// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.
//
// Modifier: Livox               dev@livoxtech.com
//
// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include <omp.h>
#include <mutex>
#include <math.h>
#include <cmath>
#include <thread>
#include <fstream>
#include <csignal>
#include <chrono>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <rclcpp/rclcpp.hpp>
#include <Eigen/Core>
#include "IMU_Processing.hpp"
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>   // [Ceiling clip] add
#include <pcl/filters/radius_outlier_removal.h> // [Denoise] add
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <livox_ros_driver2/msg/custom_msg.hpp>
#include "preprocess.h"
#include <ikd-Tree/ikd_Tree.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/time.h>
#include <algorithm> // std::max, std::fill
#include <deque>
#include <utility>
#include <iomanip>   // std::setw
#include <iostream>
#include <builtin_interfaces/msg/time.hpp>
#include <optional>
#include <limits>     // [Ceiling clip] add

#define INIT_TIME (0.1)
#define LASER_POINT_COV (0.001)
#define MAXN (720000)
#define PUBFRAME_PERIOD (20)

/*** Time Log Variables ***/
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];
double match_time = 0, solve_time = 0, solve_const_H_time = 0;
int kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
bool runtime_pos_log = false, pcd_save_en = false, time_sync_en = false, extrinsic_est_en = true, path_en = true;
/**************************/

float res_last[100000] = {0.0};
float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;
double time_diff_lidar_to_imu = 0.0;

mutex mtx_buffer;
condition_variable sig_buffer;

string root_dir = ROOT_DIR;
string map_file_path, lid_topic, imu_topic;

double res_mean_last = 0.05, total_residual = 0.0;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
int effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;
int iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_save_interval = -1, pcd_index = 0;
bool point_selected_surf[100000] = {0};
bool lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
bool scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;
bool is_first_lidar = true;

vector<vector<int>> pointSearchInd_surf;
vector<BoxPointType> cub_needrm;
vector<PointVector> Nearest_Points;
vector<double> extrinT(3, 0.0);
vector<double> extrinR(9, 0.0);
deque<double> time_buffer;
deque<PointCloudXYZI::Ptr> lidar_buffer;
deque<sensor_msgs::msg::Imu::ConstSharedPtr> imu_buffer;

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr _featsArray;

pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;
// === [NEW] 웹 경량 스트림용 다운샘플러 ===
pcl::VoxelGrid<PointType> downSizeFilterWeb;

KD_TREE<PointType> ikdtree;

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D euler_cur;
V3D position_last(Zero3d);
V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

/*** EKF inputs and output ***/
MeasureGroup Measures;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
state_ikfom state_point;
vect3 pos_lid;

nav_msgs::msg::Path path;
nav_msgs::msg::Odometry odomAftMapped;
geometry_msgs::msg::Quaternion geoQuat;
geometry_msgs::msg::PoseStamped msg_body_pose;

shared_ptr<Preprocess> p_pre(new Preprocess());
shared_ptr<ImuProcess> p_imu(new ImuProcess());

void SigHandle(int sig)
{
    flg_exit = true;
    std::cout << "catch sig %d" << sig << std::endl;
    sig_buffer.notify_all();
    rclcpp::shutdown();
}

inline void dump_lio_state_to_log(FILE *fp)
{
    V3D rot_ang(Log(state_point.rot.toRotationMatrix()));
    fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                            // Angle
    fprintf(fp, "%lf %lf %lf ", state_point.pos(0), state_point.pos(1), state_point.pos(2));    // Pos
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                                 // omega
    fprintf(fp, "%lf %lf %lf ", state_point.vel(0), state_point.vel(1), state_point.vel(2));    // Vel
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                                 // Acc
    fprintf(fp, "%lf %lf %lf ", state_point.bg(0), state_point.bg(1), state_point.bg(2));       // Bias_g
    fprintf(fp, "%lf %lf %lf ", state_point.ba(0), state_point.ba(1), state_point.ba(2));       // Bias_a
    fprintf(fp, "%lf %lf %lf ", state_point.grav[0], state_point.grav[1], state_point.grav[2]); // Bias_a
    fprintf(fp, "\r\n");
    fflush(fp);
}

void pointBodyToWorld_ikfom(PointType const *const pi, PointType *const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void pointBodyToWorld(PointType const *const pi, PointType *const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

template <typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void RGBpointBodyToWorld(PointType const *const pi, PointType *const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void RGBpointBodyLidarToIMU(PointType const *const pi, PointType *const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I * p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

void points_cache_collect()
{
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);
    // for (int i = 0; i < points_history.size(); i++) _featsArray->push_back(points_history[i]);
}

BoxPointType LocalMap_Points;
bool Localmap_Initialized = false;
void lasermap_fov_segment()
{
    cub_needrm.clear();
    kdtree_delete_counter = 0;
    kdtree_delete_time = 0.0;
    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);
    V3D pos_LiD = pos_lid;
    if (!Localmap_Initialized)
    {
        for (int i = 0; i < 3; i++)
        {
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++)
    {
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
            need_move = true;
    }
    if (!need_move)
        return;
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    double mov_dist = std::max(
        (cube_len - 2.0 * static_cast<double>(MOV_THRESHOLD) * static_cast<double>(DET_RANGE)) * 0.5 * 0.9,
        static_cast<double>(DET_RANGE) * (static_cast<double>(MOV_THRESHOLD) - 1.0));
    for (int i = 0; i < 3; i++)
    {
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE)
        {
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
        else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
        {
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    double delete_begin = omp_get_wtime();
    if (cub_needrm.size() > 0)
        kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
    kdtree_delete_time = omp_get_wtime() - delete_begin;
}

void standard_pcl_cbk(const sensor_msgs::msg::PointCloud2::UniquePtr msg)
{
    mtx_buffer.lock();
    scan_count++;
    double cur_time = get_time_sec(msg->header.stamp);
    double preprocess_start_time = omp_get_wtime();
    if (!is_first_lidar && cur_time < last_timestamp_lidar)
    {
        std::cerr << "lidar loop back, clear buffer" << std::endl;
        lidar_buffer.clear();
    }
    if (is_first_lidar)
    {
        is_first_lidar = false;
    }

    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(cur_time);
    last_timestamp_lidar = cur_time;
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double timediff_lidar_wrt_imu = 0.0;
bool timediff_set_flg = false;

// === [WEB SLIDING GLOBAL] 최근 프레임 N초 누적용 전역 버퍼
static std::deque<std::pair<double, PointCloudXYZI::Ptr>> g_web_frames; // (t, world-frame cloud)
static double g_web_sliding_seconds = 0.0;
static int g_web_sliding_max_frames = 30;
void livox_pcl_cbk(const livox_ros_driver2::msg::CustomMsg::UniquePtr msg)
{
    mtx_buffer.lock();
    double cur_time = get_time_sec(msg->header.stamp);
    double preprocess_start_time = omp_get_wtime();
    scan_count++;
    if (!is_first_lidar && cur_time < last_timestamp_lidar)
    {
        std::cerr << "lidar loop back, clear buffer" << std::endl;
        lidar_buffer.clear();
    }
    if (is_first_lidar)
    {
        is_first_lidar = false;
    }
    last_timestamp_lidar = cur_time;

    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty())
    {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n", last_timestamp_imu, last_timestamp_lidar);
    }

    if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
    {
        timediff_set_flg = true;
        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
        printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
    }

    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(last_timestamp_lidar);

    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void imu_cbk(const sensor_msgs::msg::Imu::UniquePtr msg_in)
{
    publish_count++;
    sensor_msgs::msg::Imu::SharedPtr msg(new sensor_msgs::msg::Imu(*msg_in));

    msg->header.stamp = get_ros_time(get_time_sec(msg_in->header.stamp) - time_diff_lidar_to_imu);
    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
    {
        msg->header.stamp =
            rclcpp::Time(timediff_lidar_wrt_imu + get_time_sec(msg_in->header.stamp));
    }

    double timestamp = get_time_sec(msg->header.stamp);

    mtx_buffer.lock();

    if (timestamp < last_timestamp_imu)
    {
        std::cerr << "lidar loop back, clear buffer" << std::endl;
        imu_buffer.clear();
    }

    last_timestamp_imu = timestamp;

    imu_buffer.push_back(msg);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double lidar_mean_scantime = 0.0;
int scan_num = 0;
bool sync_packages(MeasureGroup &meas)
{
    if (lidar_buffer.empty() || imu_buffer.empty())
    {
        return false;
    }

    /*** push a lidar scan ***/
    if (!lidar_pushed)
    {
        meas.lidar = lidar_buffer.front();
        meas.lidar_beg_time = time_buffer.front();
        if (meas.lidar->points.size() <= 1) // time too little
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            std::cerr << "Too few input point cloud!\n";
        }
        else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
        }
        else
        {
            scan_num++;
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
            lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
        }

        meas.lidar_end_time = lidar_end_time;

        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = get_time_sec(imu_buffer.front()->header.stamp);
    meas.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = get_time_sec(imu_buffer.front()->header.stamp);
        if (imu_time > lidar_end_time)
            break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

int process_increments = 0;
void map_incremental()
{
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    for (int i = 0; i < feats_down_size; i++)
    {
        /* transform to world frame */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
        /* decide if need add to map */
        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            PointType downsample_result, mid_point;
            mid_point.x = floor(feats_down_world->points[i].x / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            float dist = calc_dist(feats_down_world->points[i], mid_point);
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min)
            {
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                continue;
            }
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i++)
            {
                if (points_near.size() < NUM_MATCH_POINTS)
                    break;
                if (calc_dist(points_near[readd_i], mid_point) < dist)
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add)
                PointToAdd.push_back(feats_down_world->points[i]);
        }
        else
        {
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }

    double st_time = omp_get_wtime();
    add_point_size = ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false);
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
    kdtree_incremental_time = omp_get_wtime() - st_time;
}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI());
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());
void publish_frame_world(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull, const rclcpp::Time& t_ref)
{
    if (scan_pub_en)
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI::Ptr laserCloudWorld(
            new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&laserCloudFullRes->points[i],
                                &laserCloudWorld->points[i]);
        }

        sensor_msgs::msg::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = t_ref;
        laserCloudmsg.header.frame_id = "map";
        pubLaserCloudFull->publish(laserCloudmsg);
        // 웹 슬라이딩 윈도우 업데이트(전역)
        if (g_web_sliding_seconds > 1e-6) {
            g_web_frames.emplace_back(lidar_end_time, laserCloudWorld);
            const double t_now = lidar_end_time;
            while (!g_web_frames.empty()) {
                const double t0 = g_web_frames.front().first;
                if ((t_now - t0) > g_web_sliding_seconds) g_web_frames.pop_front(); else break;
            }
            if ((int)g_web_frames.size() > g_web_sliding_max_frames) {
                while ((int)g_web_frames.size() > g_web_sliding_max_frames) g_web_frames.pop_front();
            }
        }
        publish_count -= PUBFRAME_PERIOD;
    }
}

void publish_frame_body(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_body, const rclcpp::Time& t_ref)
{
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&feats_undistort->points[i],
                               &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::msg::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = t_ref;
    laserCloudmsg.header.frame_id = "base_link";
    pubLaserCloudFull_body->publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

void publish_effect_world(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudEffect, const rclcpp::Time& t_ref)
{
    PointCloudXYZI::Ptr laserCloudWorld(
        new PointCloudXYZI(effct_feat_num, 1));
    for (int i = 0; i < effct_feat_num; i++)
    {
        RGBpointBodyToWorld(&laserCloudOri->points[i],
                            &laserCloudWorld->points[i]);
    }
    sensor_msgs::msg::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
    laserCloudFullRes3.header.stamp = t_ref;
    laserCloudFullRes3.header.frame_id = "map";
    pubLaserCloudEffect->publish(laserCloudFullRes3);
}

// === [MOD] 누적맵 퍼블리시: 옵션화(map_accumulate_) ===
void publish_map(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudMap, bool map_accumulate_)
{
    PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
    int size = laserCloudFullRes->points.size();
    PointCloudXYZI::Ptr laserCloudWorld(
        new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyToWorld(&laserCloudFullRes->points[i],
                            &laserCloudWorld->points[i]);
    }

    if (map_accumulate_) {
        *pcl_wait_pub += *laserCloudWorld;      // 계속 누적(백엔드/저장용)
    } else {
        *pcl_wait_pub = *laserCloudWorld;       // 이번 프레임만(또는 최근 프레임으로 운영)
    }

    sensor_msgs::msg::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*pcl_wait_pub, laserCloudmsg);
    laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
    laserCloudmsg.header.frame_id = "map";
    pubLaserCloudMap->publish(laserCloudmsg);
}

void save_to_pcd()
{
    pcl::PCDWriter pcd_writer;
    pcd_writer.writeBinary(map_file_path, *pcl_wait_pub);
}

template <typename T>
void set_posestamp(T &out)
{
    out.pose.position.x = state_point.pos(0);
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);
    out.pose.orientation.x = geoQuat.x;
    out.pose.orientation.y = geoQuat.y;
    out.pose.orientation.z = geoQuat.z;
    out.pose.orientation.w = geoQuat.w;
}

void publish_odometry(const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped, std::unique_ptr<tf2_ros::TransformBroadcaster> &tf_br, const rclcpp::Time& t_ref)
{
    odomAftMapped.header.frame_id = "map";
    odomAftMapped.child_frame_id = "base_link";
    odomAftMapped.header.stamp = t_ref;
    set_posestamp(odomAftMapped.pose);
    pubOdomAftMapped->publish(odomAftMapped);
    auto P = kf.get_P();
    for (int i = 0; i < 6; i++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped.pose.covariance[i * 6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i * 6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i * 6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i * 6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i * 6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i * 6 + 5] = P(k, 2);
    }

    geometry_msgs::msg::TransformStamped trans;
    trans.header.frame_id = "odom";
    trans.header.stamp = odomAftMapped.header.stamp;
    trans.child_frame_id = "body";
    trans.transform.translation.x = odomAftMapped.pose.pose.position.x;
    trans.transform.translation.y = odomAftMapped.pose.pose.position.y;
    trans.transform.translation.z = odomAftMapped.pose.pose.position.z;
    trans.transform.rotation.w = odomAftMapped.pose.pose.orientation.w;
    trans.transform.rotation.x = odomAftMapped.pose.pose.orientation.x;
    trans.transform.rotation.y = odomAftMapped.pose.pose.orientation.y;
    trans.transform.rotation.z = odomAftMapped.pose.pose.orientation.z;
    // tf_br->sendTransform(trans);
}

void publish_path(rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath, const rclcpp::Time& t_ref)
{
    set_posestamp(msg_body_pose);
    msg_body_pose.header.stamp = t_ref;
    msg_body_pose.header.frame_id = "map";

    static int jjj = 0;
    jjj++;
    if (jjj % 10 == 0)
    {
        path.poses.push_back(msg_body_pose);
        pubPath->publish(path);
    }
}

void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    double match_start = omp_get_wtime();
    laserCloudOri->clear();
    corr_normvect->clear();
    total_residual = 0.0;

#ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body = feats_down_body->points[i];
        PointType &point_world = feats_down_world->points[i];

        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos);
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

        auto &points_near = Nearest_Points[i];

        if (ekfom_data.converge)
        {
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
        }

        if (!point_selected_surf[i])
            continue;

        VF(4)
        pabcd;
        point_selected_surf[i] = false;
        if (esti_plane(pabcd, points_near, 0.1f))
        {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

            if (s > 0.9)
            {
                point_selected_surf[i] = true;
                normvec->points[i].x = pabcd(0);
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2;
                res_last[i] = abs(pd2);
            }
        }
    }

    effct_feat_num = 0;

    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf[i])
        {
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
            corr_normvect->points[effct_feat_num] = normvec->points[i];
            total_residual += res_last[i];
            effct_feat_num++;
        }
    }

    if (effct_feat_num < 1)
    {
        ekfom_data.valid = false;
        std::cerr << "No Effective Points!" << std::endl;
        return;
    }

    res_mean_last = total_residual / effct_feat_num;
    match_time += omp_get_wtime() - match_start;
    double solve_start_ = omp_get_wtime();

    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12);
    ekfom_data.h.resize(effct_feat_num);

    for (int i = 0; i < effct_feat_num; i++)
    {
        const PointType &laser_p = laserCloudOri->points[i];
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        M3D point_crossmat;
        point_crossmat << SKEW_SYM_MATRX(point_this);

        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        V3D C(s.rot.conjugate() * norm_vec);
        V3D A(point_crossmat * C);
        if (extrinsic_est_en)
        {
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C);
            ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }

        ekfom_data.h(i) = -norm_p.intensity;
    }
    solve_time += omp_get_wtime() - solve_start_;
}

class LaserMappingNode : public rclcpp::Node
{
public:
    LaserMappingNode(const rclcpp::NodeOptions &options = rclcpp::NodeOptions()) : Node("laser_mapping", options)
    {
        // ---- params
        this->declare_parameter<bool>("publish.path_en", true);
        this->declare_parameter<bool>("publish.effect_map_en", false);
        this->declare_parameter<bool>("publish.map_en", false);
        this->declare_parameter<bool>("publish.scan_publish_en", true);
        this->declare_parameter<bool>("publish.dense_publish_en", true);
        this->declare_parameter<bool>("publish.scan_bodyframe_pub_en", true);
        this->declare_parameter<int>("max_iteration", 4);
        this->declare_parameter<string>("map_file_path", "");
        this->declare_parameter<string>("common.lid_topic", "/livox/lidar");
        this->declare_parameter<string>("common.imu_topic", "/livox/imu");
        this->declare_parameter<bool>("common.time_sync_en", false);
        this->declare_parameter<double>("common.time_offset_lidar_to_imu", 0.0);
        this->declare_parameter<double>("filter_size_corner", 0.5);
        this->declare_parameter<double>("filter_size_surf", 0.5);
        this->declare_parameter<double>("filter_size_map", 0.5);
        this->declare_parameter<double>("cube_side_length", 200.);
        this->declare_parameter<float>("mapping.det_range", 300.);
        this->declare_parameter<double>("mapping.fov_degree", 180.);
        this->declare_parameter<double>("mapping.gyr_cov", 0.1);
        this->declare_parameter<double>("mapping.acc_cov", 0.1);
        this->declare_parameter<double>("mapping.b_gyr_cov", 0.0001);
        this->declare_parameter<double>("mapping.b_acc_cov", 0.0001);
        this->declare_parameter<double>("preprocess.blind", 0.01);
        this->declare_parameter<int>("preprocess.lidar_type", AVIA);
        this->declare_parameter<int>("preprocess.scan_line", 16);
        this->declare_parameter<int>("preprocess.timestamp_unit", US);
        this->declare_parameter<int>("preprocess.scan_rate", 10);
        this->declare_parameter<int>("point_filter_num", 2);
        this->declare_parameter<bool>("feature_extract_enable", false);
        this->declare_parameter<bool>("runtime_pos_log_enable", false);
        this->declare_parameter<bool>("mapping.extrinsic_est_en", true);
        this->declare_parameter<bool>("pcd_save.pcd_save_en", true);
        this->declare_parameter<int>("pcd_save.interval", -1);
        this->declare_parameter<vector<double>>("mapping.extrinsic_T", vector<double>());
        this->declare_parameter<vector<double>>("mapping.extrinsic_R", vector<double>());

        // [Ceiling clip]
        this->declare_parameter<bool>("clip.enable", true);
        this->declare_parameter<double>("clip.z_min", -0.5);
        this->declare_parameter<double>("clip.z_max", 2.0);
        this->declare_parameter<string>("clip.mode", std::string("world")); // world|body|both|either

        // [Denoise]
        this->declare_parameter<bool>("filter.intensity_filter_en", false);
        this->declare_parameter<double>("filter.intensity_min", 0.0);
        this->declare_parameter<double>("filter.intensity_max", 255.0);
        this->declare_parameter<bool>("filter.range_filter_en", false);
        this->declare_parameter<double>("filter.range_min", 0.0);
        this->declare_parameter<double>("filter.range_max", 200.0);
        this->declare_parameter<bool>("filter.radius_outlier_en", false);
        this->declare_parameter<double>("filter.radius", 0.20);
        this->declare_parameter<int>("filter.min_neighbors", 2);

        // === [NEW] 웹 경량 스트림/누적 옵션
        this->declare_parameter<bool>("publish.map_accumulate", true); // 누적 여부
        this->declare_parameter<bool>("publish.web_map_en", true);     // /Laser_map_web on/off

        // 웹 맵 밀도/상한/주기/ROI (기본 보수적으로 낮춤)
        this->declare_parameter<double>("publish.web_voxel_leaf", 0.15);
        this->declare_parameter<int>("publish.web_max_points", 300000);
        this->declare_parameter<double>("publish.web_rate_hz", 2.0);       // 2 Hz
        this->declare_parameter<double>("publish.web_roi_radius", 0.0);    // 0이면 비활성
        this->declare_parameter<double>("publish.web_sliding_seconds", 0.0); // >0이면 최근 N초 프레임 누적 송출
        this->declare_parameter<int>("publish.web_sliding_max_frames", 30);

        // ---- get params
        this->get_parameter_or<bool>("publish.path_en", path_en, true);
        this->get_parameter_or<bool>("publish.effect_map_en", effect_pub_en, false);
        this->get_parameter_or<bool>("publish.map_en", map_pub_en, false);
        this->get_parameter_or<bool>("publish.scan_publish_en", scan_pub_en, true);
        this->get_parameter_or<bool>("publish.dense_publish_en", dense_pub_en, true);
        this->get_parameter_or<bool>("publish.scan_bodyframe_pub_en", scan_body_pub_en, true);
        this->get_parameter_or<int>("max_iteration", NUM_MAX_ITERATIONS, 4);
        this->get_parameter_or<string>("map_file_path", map_file_path, "");
        this->get_parameter_or<string>("common.lid_topic", lid_topic, "/livox/lidar");
        this->get_parameter_or<string>("common.imu_topic", imu_topic, "/livox/imu");
        this->get_parameter_or<bool>("common.time_sync_en", time_sync_en, false);
        this->get_parameter_or<double>("common.time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);
        this->get_parameter_or<double>("filter_size_corner", filter_size_corner_min, 0.5);
        this->get_parameter_or<double>("filter_size_surf", filter_size_surf_min, 0.5);
        this->get_parameter_or<double>("filter_size_map", filter_size_map_min, 0.5);
        this->get_parameter_or<double>("cube_side_length", cube_len, 200.f);
        this->get_parameter_or<float>("mapping.det_range", DET_RANGE, 300.f);
        this->get_parameter_or<double>("mapping.fov_degree", fov_deg, 180.f);
        this->get_parameter_or<double>("mapping.gyr_cov", gyr_cov, 0.1);
        this->get_parameter_or<double>("mapping.acc_cov", acc_cov, 0.1);
        this->get_parameter_or<double>("mapping.b_gyr_cov", b_gyr_cov, 0.0001);
        this->get_parameter_or<double>("mapping.b_acc_cov", b_acc_cov, 0.0001);
        this->get_parameter_or<double>("preprocess.blind", p_pre->blind, 0.01);
        this->get_parameter_or<int>("preprocess.lidar_type", p_pre->lidar_type, AVIA);
        this->get_parameter_or<int>("preprocess.scan_line", p_pre->N_SCANS, 16);
        this->get_parameter_or<int>("preprocess.timestamp_unit", p_pre->time_unit, US);
        this->get_parameter_or<int>("preprocess.scan_rate", p_pre->SCAN_RATE, 10);
        this->get_parameter_or<int>("point_filter_num", p_pre->point_filter_num, 2);
        this->get_parameter_or<bool>("feature_extract_enable", p_pre->feature_enabled, false);
        this->get_parameter_or<bool>("runtime_pos_log_enable", runtime_pos_log, 0);
        this->get_parameter_or<bool>("mapping.extrinsic_est_en", extrinsic_est_en, true);
        this->get_parameter_or<bool>("pcd_save.pcd_save_en", pcd_save_en, false);
        this->get_parameter_or<int>("pcd_save.interval", pcd_save_interval, -1);
        this->get_parameter_or<vector<double>>("mapping.extrinsic_T", extrinT, vector<double>());
        this->get_parameter_or<vector<double>>("mapping.extrinsic_R", extrinR, vector<double>());

        // [Ceiling clip] load
        this->get_parameter_or<bool>("clip.enable", clip_enable, true);
        this->get_parameter_or<double>("clip.z_min", clip_z_min, -0.5);
        this->get_parameter_or<double>("clip.z_max", clip_z_max, 2.0);
        this->get_parameter<std::string>("clip.mode", clip_mode);

        // [Denoise] get params
        this->get_parameter_or<bool>("filter.intensity_filter_en", intensity_filter_en, false);
        this->get_parameter_or<double>("filter.intensity_min", intensity_min, 0.0);
        this->get_parameter_or<double>("filter.intensity_max", intensity_max, 255.0);
        this->get_parameter_or<bool>("filter.range_filter_en", range_filter_en, false);
        this->get_parameter_or<double>("filter.range_min", range_min, 0.0);
        this->get_parameter_or<double>("filter.range_max", range_max, 200.0);
        this->get_parameter_or<bool>("filter.radius_outlier_en", radius_outlier_en, false);
        this->get_parameter_or<double>("filter.radius", radius_outlier_radius, 0.20);
        this->get_parameter_or<int>("filter.min_neighbors", radius_outlier_min_neighbors, 2);

        // === [NEW] 웹 스트림/옵션 값
        this->get_parameter_or<bool>("publish.map_accumulate", map_accumulate_, true);
        this->get_parameter_or<bool>("publish.web_map_en", web_map_en_, true);
        this->get_parameter_or<double>("publish.web_voxel_leaf", web_voxel_leaf_, 0.15);
        this->get_parameter_or<int>("publish.web_max_points", web_max_points_, 300000);
        this->get_parameter_or<double>("publish.web_rate_hz", web_rate_hz_, 2.0);
        this->get_parameter_or<double>("publish.web_roi_radius", web_roi_radius_, 0.0);
        this->get_parameter_or<double>("publish.web_sliding_seconds", web_sliding_seconds_, 0.0);
        this->get_parameter_or<int>("publish.web_sliding_max_frames", web_sliding_max_frames_, 30);

        RCLCPP_INFO(this->get_logger(), "p_pre->lidar_type %d", p_pre->lidar_type);

        path.header.stamp = this->get_clock()->now();
        path.header.frame_id = "map";

        FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
        HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);

        _featsArray.reset(new PointCloudXYZI());

        std::fill_n(point_selected_surf, 100000, true);
        std::fill_n(res_last, 100000, -1000.0f);
        downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
        downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
        // === [NEW]
        downSizeFilterWeb.setLeafSize(web_voxel_leaf_, web_voxel_leaf_, web_voxel_leaf_);

        Lidar_T_wrt_IMU << VEC_FROM_ARRAY(extrinT);
        Lidar_R_wrt_IMU << MAT_FROM_ARRAY(extrinR);
        p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
        p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
        p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
        p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
        p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));

        std::fill(std::begin(epsi), std::end(epsi), 0.001);
        kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);

        string pos_log_dir = root_dir + "/Log/pos_log.txt";
        fp = fopen(pos_log_dir.c_str(), "w");

        fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"), ios::out);
        fout_out.open(DEBUG_FILE_DIR("mat_out.txt"), ios::out);
        fout_dbg.open(DEBUG_FILE_DIR("dbg.txt"), ios::out);
        if (fout_pre && fout_out)
            cout << "~~~~" << ROOT_DIR << " file opened" << endl;
        else
            cout << "~~~~" << ROOT_DIR << " doesn't exist" << endl;

        if (p_pre->lidar_type == AVIA)
        {
            sub_pcl_livox_ = this->create_subscription<livox_ros_driver2::msg::CustomMsg>(lid_topic, 20, livox_pcl_cbk);
        }
        else
        {
            sub_pcl_pc_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(lid_topic, rclcpp::SensorDataQoS(), standard_pcl_cbk);
        }
        sub_imu_ = this->create_subscription<sensor_msgs::msg::Imu>(imu_topic, 10, imu_cbk);
        pubLaserCloudFull_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered", 20);
        pubLaserCloudFull_body_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered_body", 20);
        pubLaserCloudEffect_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_effected", 20);
        pubLaserCloudMap_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/Laser_map", 20);
        pubOdomAftMapped_ = this->create_publisher<nav_msgs::msg::Odometry>("/Odometry", 20);
        pubPath_ = this->create_publisher<nav_msgs::msg::Path>("/path", 20);
        // === [NEW] 웹 전용 경량 토픽 (BestEffort + depth1)
        pubLaserCloudMapWeb_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/Laser_map_web", rclcpp::QoS(rclcpp::KeepLast(1)).best_effort());

        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

        // 메인 처리 타이머(100Hz)
        auto period_ms = std::chrono::milliseconds(static_cast<int64_t>(1000.0 / 100.0));
        timer_ = rclcpp::create_timer(this, this->get_clock(), period_ms, std::bind(&LaserMappingNode::timer_callback, this));

        // 누적 맵 퍼블리시 주기(1Hz)
        auto map_period_ms = std::chrono::milliseconds(static_cast<int64_t>(1000.0));
        map_pub_timer_ = rclcpp::create_timer(this, this->get_clock(), map_period_ms, std::bind(&LaserMappingNode::map_publish_callback, this));

        // === [NEW] 웹 경량 스트림 퍼블리시 주기(5Hz)
        // 웹 퍼블리시 주기 설정
        double hz = (web_rate_hz_ > 0.1 ? web_rate_hz_ : 2.0);
        auto web_period_ms = std::chrono::milliseconds((int)std::round(1000.0 / hz));
        web_pub_timer_ = rclcpp::create_timer(this, this->get_clock(), web_period_ms, std::bind(&LaserMappingNode::web_map_publish_callback, this));

        map_save_srv_ = this->create_service<std_srvs::srv::Trigger>("map_save", std::bind(&LaserMappingNode::map_save_callback, this, std::placeholders::_1, std::placeholders::_2));

        // 전역 슬라이딩 파라미터 설정
        g_web_sliding_seconds = web_sliding_seconds_;
        g_web_sliding_max_frames = web_sliding_max_frames_;

        RCLCPP_INFO(this->get_logger(), "Node init finished. web_map_en=%d accumulate=%d web_leaf=%.2f web_max_pts=%d web_rate=%.2fHz web_roi=%.2fm slide=%.1fs (max %d frames)",
                    (int)web_map_en_, (int)map_accumulate_, web_voxel_leaf_, web_max_points_, hz, web_roi_radius_, web_sliding_seconds_, web_sliding_max_frames_);
    }

    ~LaserMappingNode()
    {
        fout_out.close();
        fout_pre.close();
        fclose(fp);
    }

private:
    inline rclcpp::Time as_node_time(const rclcpp::Time &src) const
    {
        return rclcpp::Time(src.nanoseconds(), this->get_clock()->get_clock_type());
    }
    inline rclcpp::Time as_node_time(const builtin_interfaces::msg::Time &src) const
    {
        return rclcpp::Time(src, this->get_clock()->get_clock_type());
    }

    void publish_map_to_odom(const rclcpp::Time& t_ref)
    {
        // 0) odom->base_link 최신 여부 확인 (예: 200ms 이내 응답 기다림)
      const rclcpp::Duration timeout = tf2::durationFromSec(0.2);
      if (!tf_buffer_->canTransform("odom", "base_link", t_ref, timeout)) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                             "No odom->base_link TF yet. Skipping this cycle.");
        return;
      }

      geometry_msgs::msg::TransformStamped T_odom_base;
      try {
        // 1) 최신 TF 조회
        T_odom_base = tf_buffer_->lookupTransform("odom", "base_link", t_ref);
      } catch (const tf2::TransformException &ex) {
        // 1-FAIL) 이전 성공값으로 1회 버팀(있다면)
        if (last_T_map_odom_) {
          tf_broadcaster_->sendTransform(*last_T_map_odom_);
          RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                               "lookupTransform failed (%s). Re-published last map->odom.", ex.what());
        } else {
          RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                               "lookupTransform failed (%s). No fallback TF yet.", ex.what());
        }
        return; // 이번 주기 종료
      }

      // 2) 사용 시각(used_t): odom->base_link의 stamp
      //rclcpp::Time used_t = as_node_time(T_odom_base.header.stamp);
      rclcpp::Time used_t = as_node_time(t_ref);

      // (선택) 너무 오래된 TF면 스킵
      if ((get_clock()->now() - used_t) > rclcpp::Duration::from_seconds(0.5)) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                             "odom->base_link is stale by %.3fs, skipping.",
                             (get_clock()->now() - used_t).seconds());
        return;
      }

      // 3) map->base_link (LOAM 포즈를 used_t로 찍음)
      geometry_msgs::msg::TransformStamped T_map_base;
      T_map_base.header.stamp = used_t;
      T_map_base.header.frame_id = "map";
      T_map_base.child_frame_id = "base_link";
      T_map_base.transform.translation.x = state_point.pos(0);
      T_map_base.transform.translation.y = state_point.pos(1);
      T_map_base.transform.translation.z = state_point.pos(2);
      T_map_base.transform.rotation = geoQuat;

      // 4) map->odom = (map->base) * (odom->base)^-1
      tf2::Transform map_base_tf, odom_base_tf;
      tf2::fromMsg(T_map_base.transform, map_base_tf);
      tf2::fromMsg(T_odom_base.transform, odom_base_tf);
      tf2::Transform map_odom_tf = map_base_tf * odom_base_tf.inverse();

      geometry_msgs::msg::TransformStamped T_map_odom;
      T_map_odom.header.stamp = used_t;
      T_map_odom.header.frame_id = "map";
      T_map_odom.child_frame_id = "odom";
      T_map_odom.transform = tf2::toMsg(map_odom_tf);

      tf_broadcaster_->sendTransform(T_map_odom);
      last_T_map_odom_ = T_map_odom;  // 폴백용 캐시

      // 5) Odometry도 같은 시간으로!
      odomAftMapped.header.stamp = used_t;
    }

    void timer_callback()
    {
        if (sync_packages(Measures))
        {
            if (flg_first_scan)
            {
                first_lidar_time = Measures.lidar_beg_time;
                p_imu->first_lidar_time = first_lidar_time;
                flg_first_scan = false;
                return;
            }

            double t0, t1, t2, t3, t4, t5, match_start, solve_start, svd_time;

            match_time = 0;
            kdtree_search_time = 0.0;
            solve_time = 0;
            solve_const_H_time = 0;
            svd_time = 0;
            t0 = omp_get_wtime();

            // IMU 보정 및 undistort
            p_imu->Process(Measures, kf, feats_undistort);
            state_point = kf.get_x();
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

            // ===== [DEBUG] 클리핑 전 z 범위 확인 =====
            if (feats_undistort && !feats_undistort->empty()) {
                double minz = 1e9, maxz = -1e9;
                for (const auto& p : feats_undistort->points) {
                    if (p.z < minz) minz = p.z;
                    if (p.z > maxz) maxz = p.z;
                }
                // world z 범위도 간단히 샘플링
                double wmin = 1e9, wmax = -1e9;
                for (size_t si = 0; si < feats_undistort->points.size(); si += std::max<size_t>(1, feats_undistort->points.size()/500)) {
                    const auto& p = feats_undistort->points[si];
                    V3D p_body(p.x, p.y, p.z);
                    V3D p_world = state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos;
                    const double zw = p_world(2);
                    if (zw < wmin) wmin = zw; if (zw > wmax) wmax = zw;
                }
                RCLCPP_DEBUG(this->get_logger(),
                        "[clip debug] BODY z:[%.3f, %.3f] WORLD z:[%.3f, %.3f] using[%.3f, %.3f] enable=%d mode=%s",
                        minz, maxz, wmin, wmax, clip_z_min, clip_z_max, clip_enable, clip_mode.c_str());
            }

            // [Ceiling clip] 모드별 Z 필터
            if (clip_enable && feats_undistort && !feats_undistort->empty())
            {
                PointCloudXYZI::Ptr kept(new PointCloudXYZI());
                kept->reserve(feats_undistort->size());

                size_t removed = 0;
                for (const auto& pt : feats_undistort->points) {
                    bool ok_body = true, ok_world = true;
                    // BODY z
                    ok_body = (pt.z >= clip_z_min && pt.z <= clip_z_max);
                    // WORLD z
                    V3D p_body(pt.x, pt.y, pt.z);
                    V3D p_world = state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos;
                    double zw = p_world(2);
                    ok_world = (zw >= clip_z_min && zw <= clip_z_max);

                    bool keep = false;
                    if (clip_mode == "world") keep = ok_world;
                    else if (clip_mode == "body") keep = ok_body;
                    else if (clip_mode == "both") keep = (ok_body && ok_world);
                    else /* either */ keep = (ok_body || ok_world);

                    if (keep) kept->push_back(pt); else removed++;
                }
                feats_undistort.swap(kept);
                RCLCPP_DEBUG(this->get_logger(), "[clip %s] kept=%zu removed=%zu (z=[%.2f, %.2f])",
                    clip_mode.c_str(), feats_undistort->size(), removed, clip_z_min, clip_z_max);
            }

            // [Denoise] Intensity / Range
            if ((intensity_filter_en || range_filter_en) && feats_undistort && !feats_undistort->empty())
            {
                PointCloudXYZI::Ptr kept(new PointCloudXYZI());
                kept->reserve(feats_undistort->size());
                size_t removed = 0;
                for (const auto &pt : feats_undistort->points)
                {
                    bool keep = true;
                    if (intensity_filter_en)
                    {
                        const double inten = static_cast<double>(pt.intensity);
                        if (!(inten >= intensity_min && inten <= intensity_max))
                            keep = false;
                    }
                    if (range_filter_en)
                    {
                        const double r = std::sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
                        if (!(r >= range_min && r <= range_max))
                            keep = false;
                    }
                    if (keep)
                        kept->push_back(pt);
                    else
                        removed++;
                }
                feats_undistort.swap(kept);
                RCLCPP_DEBUG(this->get_logger(), "[denoise] intensity/range kept=%zu removed=%zu (inten=[%.1f, %.1f], range=[%.2f, %.2f])",
                            feats_undistort->size(), removed, intensity_min, intensity_max, range_min, range_max);
            }

            // [Denoise] Radius outlier
            if (radius_outlier_en && feats_undistort && !feats_undistort->empty())
            {
                pcl::RadiusOutlierRemoval<PointType> ror;
                ror.setInputCloud(feats_undistort);
                ror.setRadiusSearch(radius_outlier_radius);
                ror.setMinNeighborsInRadius(radius_outlier_min_neighbors);
                PointCloudXYZI::Ptr filtered(new PointCloudXYZI());
                ror.filter(*filtered);
                size_t removed = feats_undistort->size() - filtered->size();
                feats_undistort.swap(filtered);
                RCLCPP_INFO(this->get_logger(), "[denoise] ROR kept=%zu removed=%zu (r=%.2f, k=%d)",
                            feats_undistort->size(), removed, radius_outlier_radius, radius_outlier_min_neighbors);
            }

            if (!feats_undistort || feats_undistort->empty())
            {
                RCLCPP_WARN(this->get_logger(), "No point after filtering, skip this scan!");
                return;
            }

            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? false : true;
            lasermap_fov_segment();

            // 다운샘플
            downSizeFilterSurf.setInputCloud(feats_undistort);
            downSizeFilterSurf.filter(*feats_down_body);
            t1 = omp_get_wtime();
            feats_down_size = feats_down_body->points.size();

            // kdtree 초기화
            if (ikdtree.Root_Node == nullptr)
            {
                RCLCPP_INFO(this->get_logger(), "Initialize the map kdtree");
                if (feats_down_size > 5)
                {
                    ikdtree.set_downsample_param(filter_size_map_min);
                    feats_down_world->resize(feats_down_size);
                    for (int i = 0; i < feats_down_size; i++)
                    {
                        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                    }
                    ikdtree.Build(feats_down_world->points);
                }
                return;
            }
            int featsFromMapNum = ikdtree.validnum();
            kdtree_size_st = ikdtree.size();

            if (feats_down_size < 5)
            {
                RCLCPP_WARN(this->get_logger(), "Too few points, skip this scan!");
                return;
            }

            normvec->resize(feats_down_size);
            feats_down_world->resize(feats_down_size);

            V3D ext_euler = SO3ToEuler(state_point.offset_R_L_I);
            fout_pre << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_point.pos.transpose() << " " << ext_euler.transpose() << " " << state_point.offset_T_L_I.transpose() << " " << state_point.vel.transpose()
                     << " " << state_point.bg.transpose() << " " << state_point.ba.transpose() << " " << state_point.grav << endl;

            pointSearchInd_surf.resize(feats_down_size);
            Nearest_Points.resize(feats_down_size);

            t2 = omp_get_wtime();

            // Iterated update
            double t_update_start = omp_get_wtime();
            double solve_H_time = 0;
            kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
            state_point = kf.get_x();
            euler_cur = SO3ToEuler(state_point.rot);
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
            geoQuat.x = state_point.rot.coeffs()[0];
            geoQuat.y = state_point.rot.coeffs()[1];
            geoQuat.z = state_point.rot.coeffs()[2];
            geoQuat.w = state_point.rot.coeffs()[3];

            double t_update_end = omp_get_wtime();


            // ★ 기준 시각(모든 메시지/TF에 동일 적용)
            rclcpp::Time t_ref = as_node_time(get_ros_time(lidar_end_time));

            // ★ TF(map->odom)를 먼저, t_ref로 송출
            publish_map_to_odom(t_ref);

            // ★ 같은 t_ref로 오도메트리/클라우드/패스 송출
            publish_odometry(pubOdomAftMapped_, tf_broadcaster_, t_ref);

            // map 업데이트
            t3 = omp_get_wtime();
            map_incremental();
            t5 = omp_get_wtime();

            //if (path_en)
            //    publish_path(pubPath_);
            //if (scan_pub_en)
            //    publish_frame_world(pubLaserCloudFull_);
            //if (scan_pub_en && scan_body_pub_en)
            //   publish_frame_body(pubLaserCloudFull_body_);
            //if (effect_pub_en)
            //    publish_effect_world(pubLaserCloudEffect_);

            if (path_en) publish_path(pubPath_, t_ref);
            if (scan_pub_en) publish_frame_world(pubLaserCloudFull_, t_ref);
            if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFull_body_, t_ref);
            if (effect_pub_en) publish_effect_world(pubLaserCloudEffect_, t_ref);

            if (runtime_pos_log)
            {
                frame_num++;
                kdtree_size_end = ikdtree.size();
                aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
                aver_time_icp = aver_time_icp * (frame_num - 1) / frame_num + (t_update_end - t_update_start) / frame_num;
                aver_time_match = aver_time_match * (frame_num - 1) / frame_num + (match_time) / frame_num;
                aver_time_incre = aver_time_incre * (frame_num - 1) / frame_num + (kdtree_incremental_time) / frame_num;
                aver_time_solve = aver_time_solve * (frame_num - 1) / frame_num + (solve_time + solve_H_time) / frame_num;
                aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1) / frame_num + solve_time / frame_num;
                T1[time_log_counter] = Measures.lidar_beg_time;
                s_plot[time_log_counter] = t5 - t0;
                s_plot2[time_log_counter] = feats_undistort->points.size();
                s_plot3[time_log_counter] = kdtree_incremental_time;
                s_plot4[time_log_counter] = kdtree_search_time;
                s_plot5[time_log_counter] = kdtree_delete_counter;
                s_plot6[time_log_counter] = kdtree_delete_time;
                s_plot7[time_log_counter] = kdtree_size_st;
                s_plot8[time_log_counter] = kdtree_size_end;
                s_plot9[time_log_counter] = aver_time_consu;
                s_plot10[time_log_counter] = add_point_size;
                time_log_counter++;
                printf("[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f construct H: %0.6f \n", t1 - t0, aver_time_match, aver_time_solve, t3 - t1, t5 - t3, aver_time_consu, aver_time_icp, aver_time_const_H_time);
                ext_euler = SO3ToEuler(state_point.offset_R_L_I);
                fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_point.pos.transpose() << " " << ext_euler.transpose() << " " << state_point.offset_T_L_I.transpose() << " " << state_point.vel.transpose()
                         << " " << state_point.bg.transpose() << " " << state_point.ba.transpose() << " " << state_point.grav << " " << feats_undistort->points.size() << endl;
                dump_lio_state_to_log(fp);
            }
        }
    }

    void map_publish_callback()
    {
        if (map_pub_en)
            publish_map(pubLaserCloudMap_, map_accumulate_);
    }

    // === [NEW] 웹 경량 스트림 퍼블리시 ===
    void web_map_publish_callback()
{
    if (!web_map_en_) return;

    // 0) 소스 포인트클라우드 선정
    PointCloudXYZI::Ptr src(new PointCloudXYZI());
    if (g_web_sliding_seconds > 1e-6 && !g_web_frames.empty()) {
        // 최근 N초 프레임들을 합쳐서 사용
        size_t total = 0;
        for (const auto &pr : g_web_frames) total += pr.second->size();
        src->reserve(total);
        for (const auto &pr : g_web_frames) {
            *src += *(pr.second);
        }
    } else if (pcl_wait_pub && !pcl_wait_pub->empty()) {
        // 누적맵 사용
        *src = *pcl_wait_pub;
    } else if (feats_undistort && !feats_undistort->empty()) {
        // 누적맵 비어있으면 "현재 프레임"을 월드좌표로 변환해 사용
        src->resize(feats_undistort->size());
        for (size_t i = 0; i < feats_undistort->size(); ++i) {
            RGBpointBodyToWorld(&feats_undistort->points[i], &src->points[i]);
        }
    } else {
        return; // 보낼 게 없으면 종료
    }

    // 1) ROI (반경) 필터링: 설정된 경우에만 적용
    if (web_roi_radius_ > 1e-6) {
        const double cx = state_point.pos(0);
        const double cy = state_point.pos(1);
        const double cz = state_point.pos(2);
        const double r2 = web_roi_radius_ * web_roi_radius_;
        PointCloudXYZI::Ptr roi(new PointCloudXYZI());
        roi->reserve(src->size());
        for (const auto &pt : src->points) {
            const double dx = pt.x - cx;
            const double dy = pt.y - cy;
            const double dz = pt.z - cz;
            if ((dx*dx + dy*dy + dz*dz) <= r2) {
                roi->push_back(pt);
            }
        }
        src.swap(roi);
    }

    // 2) 웹 전용 다운샘플 (leaf==0이면 생략)
    PointCloudXYZI::Ptr ds = src;
    if (web_voxel_leaf_ > 1e-6) {
        PointCloudXYZI::Ptr tmp(new PointCloudXYZI());
        downSizeFilterWeb.setInputCloud(src);
        downSizeFilterWeb.filter(*tmp);
        ds.swap(tmp);
    }

    // 3) 상한 cap
    if ((int)ds->size() > web_max_points_) ds->resize(web_max_points_);

    // 4) XYZ-only PointCloud2로 송출
    sensor_msgs::msg::PointCloud2 msg;
    msg.header.stamp = get_ros_time(lidar_end_time);
    msg.header.frame_id = "map";
    msg.height = 1;
    msg.width = ds->size();
    msg.is_bigendian = false;
    msg.is_dense = true;

    msg.fields.resize(3);
    msg.fields[0].name = "x"; msg.fields[0].offset = 0;  msg.fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32; msg.fields[0].count = 1;
    msg.fields[1].name = "y"; msg.fields[1].offset = 4;  msg.fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32; msg.fields[1].count = 1;
    msg.fields[2].name = "z"; msg.fields[2].offset = 8;  msg.fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32; msg.fields[2].count = 1;

    msg.point_step = 12;
    msg.row_step = msg.point_step * msg.width;
    msg.data.resize(msg.row_step);

    uint8_t* p = msg.data.data();
    for (const auto& pt : ds->points) {
        *reinterpret_cast<float*>(p + 0) = pt.x;
        *reinterpret_cast<float*>(p + 4) = pt.y;
        *reinterpret_cast<float*>(p + 8) = pt.z;
        p += 12;
    }

    pubLaserCloudMapWeb_->publish(msg);
}

    void map_save_callback(std_srvs::srv::Trigger::Request::ConstSharedPtr, std_srvs::srv::Trigger::Response::SharedPtr res)
    {
        RCLCPP_INFO(this->get_logger(), "Saving map to %s...", map_file_path.c_str());
        if (pcd_save_en)
        {
            save_to_pcd();
            res->success = true;
            res->message = "Map saved.";
        }
        else
        {
            res->success = false;
            res->message = "Map save disabled.";
        }
    }

private:
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_body_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudEffect_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudMap_;
    // === [NEW]
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudMapWeb_;

    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pcl_pc_;
    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr sub_pcl_livox_;

    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::TimerBase::SharedPtr map_pub_timer_;
    // === [NEW]
    rclcpp::TimerBase::SharedPtr web_pub_timer_;

    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr map_save_srv_;
    std::optional<geometry_msgs::msg::TransformStamped> last_T_map_odom_;

    bool effect_pub_en = false, map_pub_en = false;
    int effect_feat_num = 0, frame_num = 0;
    double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
    bool flg_EKF_converged, EKF_stop_flg = 0;
    double epsi[23] = {0.001};

    // [Ceiling clip]
    bool clip_enable = true;
    double clip_z_min = -1.0;
    double clip_z_max = 0.5;
    std::string clip_mode = "world";

    // [Denoise]
    bool intensity_filter_en = false;
    double intensity_min = 0.0;
    double intensity_max = 255.0;
    bool range_filter_en = false;
    double range_min = 0.0;
    double range_max = 200.0;
    bool radius_outlier_en = false;
    double radius_outlier_radius = 0.20;
    int radius_outlier_min_neighbors = 2;

    // === [NEW] 옵션 멤버
    bool map_accumulate_ = true;
    bool web_map_en_ = true;
    double web_voxel_leaf_ = 0.15;   // 기본 더 희소
    int web_max_points_ = 300000;    // 상한 축소
    double web_rate_hz_ = 2.0;       // 웹 퍼블리시 주기(Hz)
    double web_roi_radius_ = 0.0;    // ROI 반경(0=off)

    // 웹용 최근 프레임 슬라이딩 윈도우
    double web_sliding_seconds_ = 0.0;
    int web_sliding_max_frames_ = 30;
    std::deque<std::pair<double, PointCloudXYZI::Ptr>> web_frames_;

    FILE *fp;
    ofstream fout_pre, fout_out, fout_dbg;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    signal(SIGINT, SigHandle);

    rclcpp::spin(std::make_shared<LaserMappingNode>());

    if (rclcpp::ok())
        rclcpp::shutdown();

    if (pcl_wait_save->size() > 0 && pcd_save_en)
    {
        string file_name = string("scans.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        cout << "current scan saved to /PCD/" << file_name << endl;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    }

    if (runtime_pos_log)
    {
        vector<double> t, s_vec, s_vec2, s_vec3, s_vec4, s_vec5;
        FILE *fp2;
        string log_dir = root_dir + "/Log/fast_lio_time_log.csv";
        fp2 = fopen(log_dir.c_str(), "w");
        fprintf(fp2, "time_stamp, total time, scan point size, incremental time, search time, delete size, delete time, tree size st, tree size end, add point size, preprocess time\n");
        for (int i = 0; i < time_log_counter; i++)
        {
            fprintf(fp2, "%0.8f,%0.8f,%d,%0.8f,%0.8f,%d,%0.8f,%d,%d,%d,%0.8f\n", T1[i], s_plot[i], int(s_plot2[i]), s_plot3[i], s_plot4[i], int(s_plot5[i]), s_plot6[i], int(s_plot7[i]), int(s_plot8[i]), int(s_plot10[i]), s_plot11[i]);
            t.push_back(T1[i]);
            s_vec.push_back(s_plot9[i]);
            s_vec2.push_back(s_plot3[i] + s_plot6[i]);
            s_vec3.push_back(s_plot4[i]);
            s_vec5.push_back(s_plot[i]);
        }
        fclose(fp2);
    }

    return 0;
}
