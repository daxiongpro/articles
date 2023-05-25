# ros 接收点云消息（python）

之前我们讲了如何使用 c++ 接收 ros msg，[参考这里](/home/daxiongpro/code/articles/2023_04/ros_receive_pointcloud_msgs/ros_receive_pointcloud_msg_c++.md)。本文介绍如何使用 python 接收 ros msg。

> ros1 和 ros2 大同小异。ros1 中使用 rospy 和 roscpp，而 ros2 中使用 rclpy 和 rclcpp。

本文使用的是 ros1 melodic 版本，接收的消息类型是 PointCloud2 。

在实际场景中，可能会同时接收多个 msg，比如在使用速腾 M1 激光雷达时，有 6 个激光雷达同时向 6 个方向发射激光；再比如，在车上左前方、中间、前向中央、左后方、后向中央、右后方个有 1 个相机，共 6 个相机。多个设备意味着 ros 同时接收多个 topic 。因此，如果是多个 topic，需要用到时间同步。

```python
import rospy
import message_filters
from sensor_msgs.msg import PointCloud2
from argparse import ArgumentParser


class PointCloudSubscriber(object):

    def __init__(self, topic):
        """

        Args:
            topic: str or List:str. 比如 '/rslidar'
        """
        self.topic = topic

    def callback(self, *msgs):
        """

        Args:
            msgs:(tuple:PointCloud2)

        Returns:

        """
        if len(msgs) == 1:
            print("机械式激光雷达 1 topic received! 长度为 ", len(msgs))
        else:
            print("固态激光雷达 6 topic received! 长度为 ", len(msgs)))


class MechanicalPointCloudSubscriber(PointCloudSubscriber):
    """
    机械式激光雷达，每帧1个topic
    """

    def __init__(self, topic_name):
        super().__init__(topic_name)
        rospy.Subscriber(self.topic, PointCloud2, self.callback, queue_size=2)


class SolidPointCloudSubscriber(PointCloudSubscriber):
    """
    固态激光雷达，每帧6个topic
    """

    def __init__(self, topic_names):
        """
        Args:
            topic_names: (List:str)多个话题名称
        """
        super().__init__(topic_names)
        subscriber_list = [message_filters.Subscriber(
            sub, PointCloud2) for sub in topic_names]
        sync = message_filters.ApproximateTimeSynchronizer(
            subscriber_list, 10, 1, allow_headerless=True)
        sync.registerCallback(self.callback)


def main():
    parser = ArgumentParser()
    parser.add_argument('--lidar-type', type=str,
                        default='mechanical', help='solid or mechanical')
    args = parser.parse_args()

    rospy.init_node("pointcloud_subscriber")

    if args.lidar_type == 'mechanical':
        MechanicalPointCloudSubscriber("/topic_name")

    elif args.lidar_type == 'solid':
        SolidPointCloudSubscriber(
            ['topic_name_1', 'topic_name_2', 'topic_name_3', 'topic_name_4', 'topic_name_5', 'topic_name_6'])

    rospy.spin()


if __name__ == '__main__':
    main()

```

这里介绍了 2 中场景：Mechanical 和 Solid，分别代表机械式激光雷达和固态激光雷达。机械式激光雷达扫描一圈，将整个 360 度场景以 1 个 topic 发出，而固态激光雷达分别以 6 个角度分别发出激光，得到 6个 topic，因此，固态激光雷达需要时间同步。

上述代码中，核心部分为 `__init__`中调用 `self.callback` 函数，在 `callback` 函数中获取到 `msg` 参数（类型为PointCloud2），然后可以对 `msg` 进行处理。

## 日期

2023/05/25：文章创作日期

2022/07：技术研究日期
