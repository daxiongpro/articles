#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

using namespace std;

class PointCloudNode : public rclcpp::Node {
 public:
  PointCloudNode() : Node("point_cloud_node") {
    this->points_sub = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/rslidar_points", 10,
        std::bind(&PointCloudNode::on_point_cloud_received, this,
                  std::placeholders::_1));
  }
  void on_point_cloud_received(
      const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    // 处理接收到的点云消息
    RCLCPP_INFO(this->get_logger(), "RECEIVED!");
  }

 private:
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr points_sub;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<PointCloudNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();

  return 0;
}