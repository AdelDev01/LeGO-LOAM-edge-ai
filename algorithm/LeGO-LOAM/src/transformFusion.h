#ifndef TRANSFORMFUSION_H
#define TRANSFORMFUSION_H

#include "lego_loam/utility.h"
#include <cstdint>

class TransformFusion : public rclcpp::Node {
 private:
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubLaserOdometry2;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subLaserOdometry;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subOdomAftMapped;

  nav_msgs::msg::Odometry laserOdometry2;
  geometry_msgs::msg::TransformStamped laserOdometryTrans;
  std::shared_ptr<tf2_ros::TransformBroadcaster> tfBroadcaster;

  float transformSum[6];
  float transformIncre[6];
  float transformMapped[6];
  float transformBefMapped[6];
  float transformAftMapped[6];
  double _e2e_sum_ms = 0.0;
  double _e2e_max_ms = 0.0;
  uint64_t _e2e_count = 0;

 public:
  TransformFusion(const std::string &name);
  ~TransformFusion();

  void transformAssociateToMap();
  void laserOdometryHandler(const nav_msgs::msg::Odometry::SharedPtr laserOdometry);
  void odomAftMappedHandler(const nav_msgs::msg::Odometry::SharedPtr odomAftMapped);
};




#endif // TRANSFORMFUSION_H