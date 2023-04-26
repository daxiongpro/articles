# ros 接收点云 msg（C++）

ros 通过 rosbag 发布 topic，如何用 C++ 代码对消息进行接收？

本文使用 ros2 foxy 版本。

```cpp
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
```

上述代码中，首先进入 main 函数，创建 node 指针的时候，进入 PointCloudNode 的构造函数。

构造函数中，通过 Node 类的 create_subscription 函数，订阅 PointCloud2 类型的 msg，订阅的 Topic 名称为 “rslidar_points"。
