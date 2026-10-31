#ifndef IMAGEPROJECTION_H
#define IMAGEPROJECTION_H

#include "lego_loam/utility.h"
#include "lego_loam/channel.h"
#include <Eigen/QR>
#include <onnxruntime_cxx_api.h>
#include <memory>

class ImageProjection : public rclcpp::Node {
 public:

  ImageProjection(const std::string &name, Channel<ProjectionOut>& output_channel);

  ~ImageProjection();

  void cloudHandler(const sensor_msgs::msg::PointCloud2::SharedPtr laserCloudMsg);

 private:
  void findStartEndAngle();
  void resetParameters();
  void projectPointCloud();
  void groundRemoval();
  void cloudSegmentation();
  void labelComponents(int row, int col);
  void publishClouds();
  void logStats(bool final_summary); // Log statistics about the segmentation process

  pcl::PointCloud<PointType>::Ptr _laser_cloud_in;

  pcl::PointCloud<PointType>::Ptr _full_cloud;
  pcl::PointCloud<PointType>::Ptr _full_info_cloud;

  pcl::PointCloud<PointType>::Ptr _ground_cloud;
  pcl::PointCloud<PointType>::Ptr _neural_ground_cloud;
  pcl::PointCloud<PointType>::Ptr _disagreement_cloud;
  pcl::PointCloud<PointType>::Ptr _segmented_cloud;
  pcl::PointCloud<PointType>::Ptr _segmented_cloud_pure;
  pcl::PointCloud<PointType>::Ptr _outlier_cloud;

  int _vertical_scans;
  int _horizontal_scans;
  float _ang_bottom;
  float _ang_resolution_X;
  float _ang_resolution_Y;
  float _segment_theta;
  int _segment_valid_point_num;
  int _segment_valid_line_num;
  int _ground_scan_index;
  float _sensor_mount_angle;
  int _frame_count = 0;
  long long _agree_valid_total = 0;  
  long long _valid_total = 0;        
  int _neural_frame_count = 0;       
  bool use_neural_ground = true; // Flag to determine whether to use neural network for ground segmentation
  bool _export_enabled = false;
  std::string _run_dir;

  Channel<ProjectionOut>& _output_channel;
  Ort::Env _onnx_env; // ONNX Runtime environment
  std::unique_ptr<Ort::Session> _onnx_session; // ONNX Runtime session for the ground segmentation model


  bool _diagnostics = false;
  bool _enable_end_to_end_timing = false; // Flag to enable end-to-end timing measurements
  double _t_ground_sum = 0.0, _t_ground_max = 0.0;
  double _t_infer_sum = 0.0, _t_infer_max = 0.0;
  long long _t_ground_n = 0, _t_infer_n = 0;
  int _processed_frames = 0;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr _sub_laser_cloud;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr _pub_full_cloud;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr _pub_full_info_cloud;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr _pub_ground_cloud;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr _pub_neural_ground_cloud;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr _pub_disagreement_cloud;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr _pub_segmented_cloud;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr _pub_segmented_cloud_pure;
  rclcpp::Publisher<cloud_msgs::msg::CloudInfo>::SharedPtr _pub_segmented_cloud_info;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr _pub_outlier_cloud;

  cloud_msgs::msg::CloudInfo _seg_msg;

  int _label_count;

  Eigen::MatrixXf _range_mat;   // range matrix for range image
  Eigen::MatrixXi _label_mat;   // label matrix for segmentaiton marking
  Eigen::Matrix<int32_t,Eigen::Dynamic,Eigen::Dynamic> _ground_mat;  // ground matrix for ground cloud marking


};



#endif  // IMAGEPROJECTION_H
