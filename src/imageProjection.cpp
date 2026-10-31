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

#include <boost/circular_buffer.hpp>
#include "imageProjection.h"
#include "cnpy.h"
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>
#include <stdexcept>
#include <lego_loam/end_to_end_timing.h>

const std::string PARAM_VERTICAL_SCANS = "laser.num_vertical_scans";
const std::string PARAM_HORIZONTAL_SCANS = "laser.num_horizontal_scans";
const std::string PARAM_ANGLE_BOTTOM = "laser.vertical_angle_bottom";
const std::string PARAM_ANGLE_TOP = "laser.vertical_angle_top";
const std::string PARAM_GROUND_INDEX = "laser.ground_scan_index";
const std::string PARAM_SENSOR_ANGLE = "laser.sensor_mount_angle";
const std::string PARAM_SEGMENT_THETA = "image_projection.segment_theta";
const std::string PARAM_SEGMENT_POINT = "image_projection.segment_valid_point_num";
const std::string PARAM_SEGMENT_LINE = "image_projection.segment_valid_line_num";
const std::string PARAM_USE_NEURAL_GROUND = "image_projection.use_neural_ground";
const std::string PARAM_EXPORT_RUN_ID = "image_projection.export_run_id";
const std::string PARAM_DIAGNOSTICS = "image_projection.diagnostics";
const std::string PARAM_ONNX_PATH = "image_projection.onnx_path";
const std::string PARAM_EXPORT_DIR = "image_projection.export_dir";

// The first frames are excluded from the timing statistics to avoid warm-up effects
// (e.g. the first ONNX Runtime inference is slower than the following ones).
const int WARMUP_FRAMES = 10;

ImageProjection::ImageProjection(const std::string &name, Channel<ProjectionOut>& output_channel)
    : Node(name),  _output_channel(output_channel),
    _onnx_env(ORT_LOGGING_LEVEL_WARNING, "ground_segmentation")  // ONNX Runtime environment
{
  _sub_laser_cloud = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/rslidar_points", 1, std::bind(&ImageProjection::cloudHandler, this, std::placeholders::_1));

  _pub_full_cloud = this->create_publisher<sensor_msgs::msg::PointCloud2>("/full_cloud_projected", 1);
  _pub_full_info_cloud = this->create_publisher<sensor_msgs::msg::PointCloud2>("/full_cloud_info", 1);
  _pub_ground_cloud = this->create_publisher<sensor_msgs::msg::PointCloud2>("/ground_cloud", 1);
  _pub_neural_ground_cloud = this->create_publisher<sensor_msgs::msg::PointCloud2>("/neural_ground_cloud", 1);
  _pub_disagreement_cloud = this->create_publisher<sensor_msgs::msg::PointCloud2>("/disagreement_cloud", 1);
  _pub_segmented_cloud = this->create_publisher<sensor_msgs::msg::PointCloud2>("/segmented_cloud", 1);
  _pub_segmented_cloud_pure = this->create_publisher<sensor_msgs::msg::PointCloud2>("/segmented_cloud_pure", 1);
  _pub_segmented_cloud_info = this->create_publisher<cloud_msgs::msg::CloudInfo>("/segmented_cloud_info", 1);
  _pub_outlier_cloud = this->create_publisher<sensor_msgs::msg::PointCloud2>("/outlier_cloud", 1);

  float vertical_angle_top;
  // Declare parameters
  this->declare_parameter(PARAM_VERTICAL_SCANS,_vertical_scans);
  this->declare_parameter(PARAM_HORIZONTAL_SCANS,_horizontal_scans);
  this->declare_parameter(PARAM_ANGLE_BOTTOM,_ang_bottom);
  this->declare_parameter(PARAM_ANGLE_TOP,vertical_angle_top);
  this->declare_parameter(PARAM_GROUND_INDEX,_ground_scan_index);
  this->declare_parameter(PARAM_SENSOR_ANGLE,_sensor_mount_angle);
  this->declare_parameter(PARAM_SEGMENT_THETA,_segment_theta);
  this->declare_parameter(PARAM_SEGMENT_POINT,_segment_valid_point_num);
  this->declare_parameter(PARAM_SEGMENT_LINE,_segment_valid_line_num);
  this->declare_parameter(PARAM_USE_NEURAL_GROUND, use_neural_ground);
  this->declare_parameter("image_projection.enable_end_to_end_timing", false);
  int export_run_id = -1;  // -1 = export disabled, N >= 0 exports to run_N
  this->declare_parameter(PARAM_EXPORT_RUN_ID, export_run_id);
  this->declare_parameter(PARAM_DIAGNOSTICS, false);
  this->declare_parameter(PARAM_ONNX_PATH, std::string(""));
  this->declare_parameter(PARAM_EXPORT_DIR, std::string(""));

  // Read parameters
  if (!this->get_parameter(PARAM_VERTICAL_SCANS, _vertical_scans)) {
    RCLCPP_WARN(this->get_logger(), "Parameter %s not found", PARAM_VERTICAL_SCANS.c_str());
  }
  if (!this->get_parameter(PARAM_HORIZONTAL_SCANS, _horizontal_scans)) {
    RCLCPP_WARN(this->get_logger(), "Parameter %s not found", PARAM_HORIZONTAL_SCANS.c_str());
  }
  if (!this->get_parameter(PARAM_ANGLE_BOTTOM, _ang_bottom)) {
    RCLCPP_WARN(this->get_logger(), "Parameter %s not found", PARAM_ANGLE_BOTTOM.c_str());
  }
  if (!this->get_parameter(PARAM_ANGLE_TOP, vertical_angle_top)) {
    RCLCPP_WARN(this->get_logger(), "Parameter %s not found", PARAM_ANGLE_TOP.c_str());
  }
  if (!this->get_parameter(PARAM_GROUND_INDEX, _ground_scan_index)) {
    RCLCPP_WARN(this->get_logger(), "Parameter %s not found", PARAM_GROUND_INDEX.c_str());
  }
  if (!this->get_parameter(PARAM_SENSOR_ANGLE, _sensor_mount_angle)) {
    RCLCPP_WARN(this->get_logger(), "Parameter %s not found", PARAM_SENSOR_ANGLE.c_str());
  }
  if (!this->get_parameter(PARAM_SEGMENT_THETA, _segment_theta)) {
    RCLCPP_WARN(this->get_logger(), "Parameter %s not found", PARAM_SEGMENT_THETA.c_str());
  }
  if (!this->get_parameter(PARAM_SEGMENT_POINT, _segment_valid_point_num)) {
    RCLCPP_WARN(this->get_logger(), "Parameter %s not found", PARAM_SEGMENT_POINT.c_str());
  }
  if (!this->get_parameter(PARAM_SEGMENT_LINE, _segment_valid_line_num)) {
    RCLCPP_WARN(this->get_logger(), "Parameter %s not found", PARAM_SEGMENT_LINE.c_str());
  }
  if (!this->get_parameter(PARAM_USE_NEURAL_GROUND, use_neural_ground)) {
    RCLCPP_WARN(this->get_logger(), "Parameter %s not found", PARAM_USE_NEURAL_GROUND.c_str());
  }
  if (!this->get_parameter(PARAM_EXPORT_RUN_ID, export_run_id)) {
    RCLCPP_WARN(this->get_logger(), "Parameter %s not found; export disabled", PARAM_EXPORT_RUN_ID.c_str());
    export_run_id = -1;
  }
  this->get_parameter("image_projection.enable_end_to_end_timing", _enable_end_to_end_timing);
  lego_loam_timing::setEnabled(_enable_end_to_end_timing);
  _export_enabled = (export_run_id >= 0);

  std::string onnx_path;
  std::string export_base_dir;
  this->get_parameter(PARAM_DIAGNOSTICS, _diagnostics);
  this->get_parameter(PARAM_ONNX_PATH, onnx_path);
  this->get_parameter(PARAM_EXPORT_DIR, export_base_dir);

  if (use_neural_ground && onnx_path.empty()) {
    RCLCPP_FATAL(this->get_logger(), "Neural ground segmentation enabled but ONNX model path is empty.");
    
    throw std::runtime_error("ONNX model path is missing.");

    RCLCPP_INFO(this->get_logger(), "Using neural network for ground segmentation.");
  } else {
    RCLCPP_INFO(this->get_logger(), "Using classical method for ground segmentation.");
  }

  // The ONNX session is created only in neural mode: the classical mode does not need the model
  if (use_neural_ground) {
    try {
      _onnx_session = std::make_unique<Ort::Session>(
          _onnx_env, onnx_path.c_str(), Ort::SessionOptions{nullptr});
      RCLCPP_INFO(this->get_logger(), "ONNX model loaded from: %s", onnx_path.c_str());
    } catch (const Ort::Exception& e) {
      RCLCPP_FATAL(this->get_logger(), "Unable to load ONNX model %s: %s",
                   onnx_path.c_str(), e.what());
      throw;
    }
  }
  RCLCPP_INFO(this->get_logger(), "Diagnostics %s", _diagnostics ? "enabled" : "disabled");

  _ang_resolution_X = (M_PI*2) / (_horizontal_scans);
  _ang_resolution_Y = DEG_TO_RAD*(vertical_angle_top - _ang_bottom) / float(_vertical_scans-1);
  _ang_bottom = -( _ang_bottom - 0.1) * DEG_TO_RAD;
  _segment_theta *= DEG_TO_RAD;
  _sensor_mount_angle *= DEG_TO_RAD;

  const size_t cloud_size = _vertical_scans * _horizontal_scans;

  _laser_cloud_in.reset(new pcl::PointCloud<PointType>());
  _full_cloud.reset(new pcl::PointCloud<PointType>());
  _full_info_cloud.reset(new pcl::PointCloud<PointType>());

  _ground_cloud.reset(new pcl::PointCloud<PointType>());
  _neural_ground_cloud.reset(new pcl::PointCloud<PointType>());
  _disagreement_cloud.reset(new pcl::PointCloud<PointType>());
  _segmented_cloud.reset(new pcl::PointCloud<PointType>());
  _segmented_cloud_pure.reset(new pcl::PointCloud<PointType>());
  _outlier_cloud.reset(new pcl::PointCloud<PointType>());

  _full_cloud->points.resize(cloud_size);
  _full_info_cloud->points.resize(cloud_size);

  // Dataset export: matrices are saved to <export_dir>/run_N, where N identifies the bag
  if (_export_enabled) {
    _run_dir = export_base_dir + "/run_" + std::to_string(export_run_id);

    bool ok = !(mkdir(export_base_dir.c_str(), 0777) != 0 && errno != EEXIST);
    ok = ok && !(mkdir(_run_dir.c_str(), 0777) != 0 && errno != EEXIST);

    if (ok) {
      RCLCPP_INFO(this->get_logger(), "Exporting matrices to: %s", _run_dir.c_str());
    } else {
      RCLCPP_ERROR(this->get_logger(), "Unable to create %s: %s. Export disabled.",
                   _run_dir.c_str(), std::strerror(errno));
      _export_enabled = false;
    }
  }

}

void ImageProjection::resetParameters() {
  const size_t cloud_size = _vertical_scans * _horizontal_scans;
  PointType nanPoint;
  nanPoint.x = std::numeric_limits<float>::quiet_NaN();
  nanPoint.y = std::numeric_limits<float>::quiet_NaN();
  nanPoint.z = std::numeric_limits<float>::quiet_NaN();

  _laser_cloud_in->clear();
  _ground_cloud->clear();
  _neural_ground_cloud->clear();
  _disagreement_cloud->clear();
  _segmented_cloud->clear();
  _segmented_cloud_pure->clear();
  _outlier_cloud->clear();

  _range_mat.resize(_vertical_scans, _horizontal_scans);
  _ground_mat.resize(_vertical_scans, _horizontal_scans);
  _label_mat.resize(_vertical_scans, _horizontal_scans);

  _range_mat.fill(FLT_MAX);
  _ground_mat.setZero();
  _label_mat.setZero();

  _label_count = 1;

  std::fill(_full_cloud->points.begin(), _full_cloud->points.end(), nanPoint);
  std::fill(_full_info_cloud->points.begin(), _full_info_cloud->points.end(),
            nanPoint);

  _seg_msg.start_ring_index.assign(_vertical_scans, 0);
  _seg_msg.end_ring_index.assign(_vertical_scans, 0);

  _seg_msg.segmented_cloud_ground_flag.assign(cloud_size, false);
  _seg_msg.segmented_cloud_col_ind.assign(cloud_size, 0);
  _seg_msg.segmented_cloud_range.assign(cloud_size, 0);
}

void ImageProjection::cloudHandler(
    const sensor_msgs::msg::PointCloud2::SharedPtr laserCloudMsg) {
  using Clock = std::chrono::steady_clock;
  auto ms_since = [](Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
  };
  // Timing statistics skip the first WARMUP_FRAMES frames
  const bool measure = (_processed_frames >= WARMUP_FRAMES);
  if (_enable_end_to_end_timing && measure) {
    lego_loam_timing::record(laserCloudMsg->header.stamp.sec, laserCloudMsg->header.stamp.nanosec, Clock::now());
  }

  // Reset parameters
  resetParameters();

  // Copy and remove NAN points
  pcl::fromROSMsg(*laserCloudMsg, *_laser_cloud_in);
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*_laser_cloud_in, *_laser_cloud_in, indices);
  _seg_msg.header = laserCloudMsg->header;

  findStartEndAngle();
  // Range image projection
  projectPointCloud();

  // The classical ground removal is needed when it is the active method, when it is
  // used as a reference for the diagnostics, or to generate the labels for the dataset export.
  // In neural mode without diagnostics it is skipped, so the measured cost is that of
  // the neuralized pipeline only.
  const bool run_classic = !use_neural_ground || _diagnostics || _export_enabled;
  if (run_classic) {
    const auto t_ground0 = Clock::now();
    groundRemoval();
    const double dt_ground = ms_since(t_ground0);
    if (measure) {
      _t_ground_sum += dt_ground;
      _t_ground_max = std::max(_t_ground_max, dt_ground);
      _t_ground_n++;
    }
  }

  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> range_rowmajor = _range_mat;

  // Dataset export (range image + classical ground labels), disabled by default
  if (_export_enabled) {
    _frame_count++;

    Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> ground_rowmajor = _ground_mat.cast<int32_t>();

    const std::string range_path = _run_dir + "/range_" + std::to_string(_frame_count) + ".npy";
    const std::string ground_path = _run_dir + "/ground_" + std::to_string(_frame_count) + ".npy";

    cnpy::npy_save(range_path, range_rowmajor.data(),
                   {(size_t)_vertical_scans, (size_t)_horizontal_scans}, "w");
    cnpy::npy_save(ground_path, ground_rowmajor.data(),
                   {(size_t)_vertical_scans, (size_t)_horizontal_scans}, "w");
  }

  if (use_neural_ground) {

    const auto t_infer0 = Clock::now();

    std::vector<int64_t> input_shape = {1, 1, (int64_t)_vertical_scans, (int64_t)_horizontal_scans};
    size_t input_size = 1 * 1 * _vertical_scans * _horizontal_scans;

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Same input normalization as in training: pixels without a return are set to 0,
    // ranges are divided by 100
    Eigen::ArrayXXf range_array = range_rowmajor.array();
    range_array = (range_array >= 1e10f).select(0.0f, range_array);
    range_array = range_array / 100.0f;
    range_rowmajor = range_array.matrix();

    std::vector<float> input_data(range_rowmajor.data(), range_rowmajor.data() + range_rowmajor.size());

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      memory_info, input_data.data(), input_size, input_shape.data(), input_shape.size()
    );

    const char* input_names[] = {"range_image"};
    const char* output_names[] = {"ground_logits"};

    auto output_tensors = _onnx_session->Run(
       Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1
    );

    float* output_data = output_tensors[0].GetTensorMutableData<float>();

    // Binary ground mask: sigmoid on the logits, then threshold at 0.5
    Eigen::MatrixXi neural_ground_mat(_vertical_scans, _horizontal_scans);

    for (int i = 0; i < _vertical_scans; i++) {
      for (int j = 0; j < _horizontal_scans; j++) {
        int index = i * _horizontal_scans + j;
        float logit = output_data[index];
        float probability = 1.0f / (1.0f + std::exp(-logit));
        neural_ground_mat(i, j) = (probability >= 0.5f) ? 1 : 0;
      }
    }

    const double dt_infer = ms_since(t_infer0);
    if (measure) {
      _t_infer_sum += dt_infer;
      _t_infer_max = std::max(_t_infer_max, dt_infer);
      _t_infer_n++;
    }
    _neural_frame_count++;

    if (_diagnostics) {
      // Agreement with the classical method, computed only on valid pixels (with a return)
      // and rows 0.._ground_scan_index, i.e. the area replaced by the network.
      // It must be computed BEFORE the replacement below, while _ground_mat still holds
      // the classical result.
      long long agree_frame = 0;
      long long valid_frame = 0;

      for (int i = 0; i <= _ground_scan_index; i++) {
        for (int j = 0; j < _horizontal_scans; j++) {
          if (_range_mat(i, j) == FLT_MAX) continue;  // pixel without a return
          valid_frame++;
          if (neural_ground_mat(i, j) == _ground_mat(i, j)) {
            agree_frame++;
          }
        }
      }

      _agree_valid_total += agree_frame;
      _valid_total += valid_frame;

      // Clouds for visualization in RViz: network ground points, and points where
      // network and classical method disagree
      for (int i = 0; i < _vertical_scans; ++i) {
        for (int j = 0; j < _horizontal_scans; ++j) {
          if (neural_ground_mat(i, j) == 1)
            _neural_ground_cloud->push_back(_full_cloud->points[j + i * _horizontal_scans]);
        }
      }

      for (int i = 0; i < _vertical_scans; ++i) {
        for (int j = 0; j < _horizontal_scans; ++j) {
          if (neural_ground_mat(i, j) != _ground_mat(i, j))
            _disagreement_cloud->push_back(_full_cloud->points[j + i * _horizontal_scans]);
        }
      }
    }

    // Replace the classical ground/label matrices with the network output.
    // Only valid pixels and rows up to _ground_scan_index can be ground, as in the
    // classical method; the downstream pipeline (segmentation, features, odometry,
    // mapping) then uses the network result.
    for (int i = 0; i < _vertical_scans; ++i) {
      for (int j = 0; j < _horizontal_scans; ++j) {
        bool valid = (_range_mat(i, j) != FLT_MAX);
        bool is_ground = (neural_ground_mat(i, j) == 1) && valid && (i <= _ground_scan_index);

        _ground_mat(i, j) = is_ground ? 1 : 0;
        _label_mat(i, j) = (is_ground || !valid) ? -1 : 0;
      }
    }

  }  // end if (use_neural_ground)

  // Point cloud segmentation
  cloudSegmentation();
  // publish (optionally)
  publishClouds();

  _processed_frames++;
  if (_processed_frames % 100 == 0) {
    logStats(false);
  }
}

void ImageProjection::logStats(bool final_summary) {
  const char* tag = final_summary ? "FINAL SUMMARY" : "Stats";

  RCLCPP_INFO(this->get_logger(), "%s: mode=%s, processed frames=%d",
              tag, use_neural_ground ? "neural" : "classical", _processed_frames);

  if (_t_ground_n > 0) {
    RCLCPP_INFO(this->get_logger(),
                "%s: classical groundRemoval mean %.3f ms, max %.3f ms (%lld frames measured)",
                tag, _t_ground_sum / _t_ground_n, _t_ground_max, _t_ground_n);
  }
  if (_t_infer_n > 0) {
    RCLCPP_INFO(this->get_logger(),
                "%s: neural inference (input + Run + threshold) mean %.3f ms, max %.3f ms (%lld frames measured)",
                tag, _t_infer_sum / _t_infer_n, _t_infer_max, _t_infer_n);
  }
  if (_valid_total > 0) {
    RCLCPP_INFO(this->get_logger(),
                "%s: network/classical agreement (valid pixels, rows 0-%d): %.3f%%",
                tag, _ground_scan_index,
                100.0 * (double)_agree_valid_total / (double)_valid_total);
  }
}

ImageProjection::~ImageProjection() {
  // Final summary printed when the node is destroyed (e.g. Ctrl+C at the end of a bag)
  if (_processed_frames > 0) {
    logStats(true);
  }
}


void ImageProjection::projectPointCloud() {
  // range image projection
  const size_t cloudSize = _laser_cloud_in->points.size();

  for (size_t i = 0; i < cloudSize; ++i) {
    PointType thisPoint = _laser_cloud_in->points[i];

    float range = sqrt(thisPoint.x * thisPoint.x +
                       thisPoint.y * thisPoint.y +
                       thisPoint.z * thisPoint.z);

    // find the row and column index in the image for this point
    float verticalAngle = std::asin(thisPoint.z / range);
        //std::atan2(thisPoint.z, sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y));

    int rowIdn = (verticalAngle + _ang_bottom) / _ang_resolution_Y;
    if (rowIdn < 0 || rowIdn >= _vertical_scans) {
      continue;
    }

    float horizonAngle = std::atan2(thisPoint.x, thisPoint.y);

    int columnIdn = -round((horizonAngle - M_PI_2) / _ang_resolution_X) + _horizontal_scans * 0.5;

    if (columnIdn >= _horizontal_scans){
      columnIdn -= _horizontal_scans;
    }

    if (columnIdn < 0 || columnIdn >= _horizontal_scans){
      continue;
    }

    if (range < 0.1){
      continue;
    }

    _range_mat(rowIdn, columnIdn) = range;

    thisPoint.intensity = (float)rowIdn + (float)columnIdn / 10000.0;

    size_t index = columnIdn + rowIdn * _horizontal_scans;
    _full_cloud->points[index] = thisPoint;
    // the corresponding range of a point is saved as "intensity"
    _full_info_cloud->points[index] = thisPoint;
    _full_info_cloud->points[index].intensity = range;
  }
}

void ImageProjection::findStartEndAngle() {
  // start and end orientation of this cloud
  auto point = _laser_cloud_in->points.front();
  _seg_msg.start_orientation = -std::atan2(point.y, point.x);

  point = _laser_cloud_in->points.back();
  _seg_msg.end_orientation = -std::atan2(point.y, point.x) + 2 * M_PI;

  if (_seg_msg.end_orientation - _seg_msg.start_orientation > 3 * M_PI) {
    _seg_msg.end_orientation -= 2 * M_PI;
  } else if (_seg_msg.end_orientation - _seg_msg.start_orientation < M_PI) {
    _seg_msg.end_orientation += 2 * M_PI;
  }
  _seg_msg.orientation_diff =
      _seg_msg.end_orientation - _seg_msg.start_orientation;
}

void ImageProjection::groundRemoval() {
  // _ground_mat
  // -1, no valid info to check if ground of not
  //  0, initial value, after validation, means not ground
  //  1, ground
  for (int j = 0; j < _horizontal_scans; ++j) {
    for (int i = 0; i < _ground_scan_index; ++i) {
      int lowerInd = j + (i)*_horizontal_scans;
      int upperInd = j + (i + 1) * _horizontal_scans;

      if (_full_cloud->points[lowerInd].intensity == -1 ||
          _full_cloud->points[upperInd].intensity == -1) {
        // no info to check, invalid points
        _ground_mat(i, j) = -1;
        continue;
      }

      float dX =
          _full_cloud->points[upperInd].x - _full_cloud->points[lowerInd].x;
      float dY =
          _full_cloud->points[upperInd].y - _full_cloud->points[lowerInd].y;
      float dZ =
          _full_cloud->points[upperInd].z - _full_cloud->points[lowerInd].z;

      float vertical_angle = std::atan2(dZ , sqrt(dX * dX + dY * dY + dZ * dZ));

      // TODO: review this change

      if ( (vertical_angle - _sensor_mount_angle) <= 10 * DEG_TO_RAD) {
        _ground_mat(i, j) = 1;
        _ground_mat(i + 1, j) = 1;
      }
    }
  }
  // extract ground cloud (_ground_mat == 1)
  // mark entry that doesn't need to label (ground and invalid point) for
  // segmentation note that ground remove is from 0~_N_scan-1, need _range_mat
  // for mark label matrix for the 16th scan
  for (int i = 0; i < _vertical_scans; ++i) {
    for (int j = 0; j < _horizontal_scans; ++j) {
      if (_ground_mat(i, j) == 1 ||
          _range_mat(i, j) == FLT_MAX) {
        _label_mat(i, j) = -1;
      }
    }
  }

  for (int i = 0; i <= _ground_scan_index; ++i) {
    for (int j = 0; j < _horizontal_scans; ++j) {
      if (_ground_mat(i, j) == 1)
        _ground_cloud->push_back(_full_cloud->points[j + i * _horizontal_scans]);
    }
  }
}

void ImageProjection::cloudSegmentation() {
  // segmentation process
  for (int i = 0; i < _vertical_scans; ++i)
    for (int j = 0; j < _horizontal_scans; ++j)
      if (_label_mat(i, j) == 0) labelComponents(i, j);

  int sizeOfSegCloud = 0;
  // extract segmented cloud for lidar odometry
  for (int i = 0; i < _vertical_scans; ++i) {
    _seg_msg.start_ring_index[i] = sizeOfSegCloud - 1 + 5;

    for (int j = 0; j < _horizontal_scans; ++j) {
      if (_label_mat(i, j) > 0 || _ground_mat(i, j) == 1) {
        // outliers that will not be used for optimization (always continue)
        if (_label_mat(i, j) == 999999) {
          if (i > _ground_scan_index && j % 5 == 0) {
            _outlier_cloud->push_back(
                _full_cloud->points[j + i * _horizontal_scans]);
            continue;
          } else {
            continue;
          }
        }
        // majority of ground points are skipped
        if (_ground_mat(i, j) == 1) {
          if (j % 5 != 0 && j > 5 && j < _horizontal_scans - 5) continue;
        }
        // mark ground points so they will not be considered as edge features
        // later
        _seg_msg.segmented_cloud_ground_flag[sizeOfSegCloud] =
            (_ground_mat(i, j) == 1);
        // mark the points' column index for marking occlusion later
        _seg_msg.segmented_cloud_col_ind[sizeOfSegCloud] = j;
        // save range info
        _seg_msg.segmented_cloud_range[sizeOfSegCloud] =
            _range_mat(i, j);
        // save seg cloud
        _segmented_cloud->push_back(_full_cloud->points[j + i * _horizontal_scans]);
        // size of seg cloud
        ++sizeOfSegCloud;
      }
    }

    _seg_msg.end_ring_index[i] = sizeOfSegCloud - 1 - 5;
  }

  // extract segmented cloud for visualization
  for (int i = 0; i < _vertical_scans; ++i) {
    for (int j = 0; j < _horizontal_scans; ++j) {
      if (_label_mat(i, j) > 0 && _label_mat(i, j) != 999999) {
        _segmented_cloud_pure->push_back(
            _full_cloud->points[j + i * _horizontal_scans]);
        _segmented_cloud_pure->points.back().intensity =
            _label_mat(i, j);
      }
    }
  }
}

void ImageProjection::labelComponents(int row, int col) {

  const float segmentThetaThreshold = tan(_segment_theta);

  std::vector<bool> lineCountFlag(_vertical_scans, false);
  const size_t cloud_size = _vertical_scans * _horizontal_scans;
  using Coord2D = Eigen::Vector2i;
  boost::circular_buffer<Coord2D> queue(cloud_size);
  boost::circular_buffer<Coord2D> all_pushed(cloud_size);

  queue.push_back({ row,col } );
  all_pushed.push_back({ row,col } );

  const Coord2D neighborIterator[4] = {
      {0, -1}, {-1, 0}, {1, 0}, {0, 1}};

  while (queue.size() > 0) {
    // Pop point
    Coord2D fromInd = queue.front();
    queue.pop_front();

    // Mark popped point
    _label_mat(fromInd.x(), fromInd.y()) = _label_count;
    // Loop through all the neighboring grids of popped grid

    for (const auto& iter : neighborIterator) {
      // new index
      int thisIndX = fromInd.x() + iter.x();
      int thisIndY = fromInd.y() + iter.y();
      // index should be within the boundary
      if (thisIndX < 0 || thisIndX >= _vertical_scans){
        continue;
      }
      // at range image margin (left or right side)
      if (thisIndY < 0){
        thisIndY = _horizontal_scans - 1;
      }
      if (thisIndY >= _horizontal_scans){
        thisIndY = 0;
      }
      // prevent infinite loop (caused by put already examined point back)
      if (_label_mat(thisIndX, thisIndY) != 0){
        continue;
      }

      float d1 = std::max(_range_mat(fromInd.x(), fromInd.y()),
                    _range_mat(thisIndX, thisIndY));
      float d2 = std::min(_range_mat(fromInd.x(), fromInd.y()),
                    _range_mat(thisIndX, thisIndY));

      float alpha = (iter.x() == 0) ? _ang_resolution_X : _ang_resolution_Y;
      float tang = (d2 * sin(alpha) / (d1 - d2 * cos(alpha)));

      if (tang > segmentThetaThreshold) {
        queue.push_back( {thisIndX, thisIndY } );

        _label_mat(thisIndX, thisIndY) = _label_count;
        lineCountFlag[thisIndX] = true;

        all_pushed.push_back(  {thisIndX, thisIndY } );
      }
    }
  }

  // check if this segment is valid
  bool feasibleSegment = false;
  if (all_pushed.size() >= 30){
    feasibleSegment = true;
  }
  else if (static_cast<int>(all_pushed.size()) >= _segment_valid_point_num) {
    int lineCount = 0;
    for (int i = 0; i < _vertical_scans; ++i) {
      if (lineCountFlag[i] == true) ++lineCount;
    }
    if (lineCount >= _segment_valid_line_num) feasibleSegment = true;
  }
  // segment is valid, mark these points
  if (feasibleSegment == true) {
    ++_label_count;
  } else {  // segment is invalid, mark these points
    for (size_t i = 0; i < all_pushed.size(); ++i) {
      _label_mat(all_pushed[i].x(), all_pushed[i].y()) = 999999;
    }
  }
}

void ImageProjection::publishClouds() {

  sensor_msgs::msg::PointCloud2 temp;
  temp.header.stamp = _seg_msg.header.stamp;
  temp.header.frame_id = "base_link";

  auto PublishCloud = [](rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub, sensor_msgs::msg::PointCloud2& temp,
                          const pcl::PointCloud<PointType>::Ptr& cloud) {
    if (pub->get_subscription_count() != 0) {
       sensor_msgs::msg::PointCloud2 _temp;
      pcl::toROSMsg(*cloud, _temp);
      _temp.header.stamp = temp.header.stamp;
      _temp.header.frame_id = temp.header.frame_id;
      pub->publish(_temp);
    }
  };

  PublishCloud(_pub_outlier_cloud, temp, _outlier_cloud);
  PublishCloud(_pub_segmented_cloud, temp, _segmented_cloud);
  PublishCloud(_pub_full_cloud, temp, _full_cloud);
  PublishCloud(_pub_ground_cloud, temp, _ground_cloud);
  PublishCloud(_pub_neural_ground_cloud, temp, _neural_ground_cloud);  // network ground points (diagnostics only)
  PublishCloud(_pub_disagreement_cloud, temp, _disagreement_cloud);  // network/classical disagreement (diagnostics only)
  PublishCloud(_pub_segmented_cloud_pure, temp, _segmented_cloud_pure);
  PublishCloud(_pub_full_info_cloud, temp, _full_info_cloud);

  if (_pub_segmented_cloud_info->get_subscription_count() != 0) {
    _pub_segmented_cloud_info->publish(_seg_msg);
  }

  //--------------------
  ProjectionOut out;
  out.outlier_cloud.reset(new pcl::PointCloud<PointType>());
  out.segmented_cloud.reset(new pcl::PointCloud<PointType>());

  std::swap( out.seg_msg, _seg_msg);
  std::swap(out.outlier_cloud, _outlier_cloud);
  std::swap(out.segmented_cloud, _segmented_cloud);

  _output_channel.send( std::move(out) );

}