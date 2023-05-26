# ros1 发布 bbox 话题

ros 发布 bbox 有两种方式：

* 使用 jsk_recognition_msgs.msg 插件
* 自定义 MarkerArray

## 使用 jsk_recognition_msgs.msg 插件

这种方法先要安装 jsk 插件，可以用 pip 安装，具体方法请百度。使用 jsk 插件发布 bbox 代码如下：

```python
import rospy
from jsk_recognition_msgs.msg import BoundingBoxArray, BoundingBox
from pyquaternion import Quaternion


class JskArrBBPub(rospy.Publisher):
    """
    jsk_bbox plugins
    these codes are based on CenterPoint
    """

    def __init__(self, name='pp_boxes', frame_id='rslidar'):
        super().__init__(name, BoundingBoxArray, queue_size=2)  # ####
        self.name = name
        self.frame_id = frame_id

    def _yaw2quaternion(yaw: float) -> Quaternion:
        return Quaternion(axis=[0, 0, 1], radians=yaw)

    def publish(self, scores_3d, labels_3d, boxes_3d):

        arr_bbox = BoundingBoxArray()
        for i in range(boxes_3d.shape[0]):
            bbox = BoundingBox()
            bbox.header.frame_id = self.frame_id
            bbox.header.stamp = rospy.Time.now()
            q = self._yaw2quaternion(float(boxes_3d[i][6]))
            bbox.pose.orientation.x = q[1]
            bbox.pose.orientation.y = q[2]
            bbox.pose.orientation.z = q[3]
            bbox.pose.orientation.w = q[0]
            bbox.pose.position.x = float(boxes_3d[i][0])
            bbox.pose.position.y = float(boxes_3d[i][1])
            bbox.pose.position.z = float(boxes_3d[i][2])
            bbox.dimensions.x = float(boxes_3d[i][3])
            bbox.dimensions.y = float(boxes_3d[i][4])
            bbox.dimensions.z = float(boxes_3d[i][5])
            bbox.value = scores_3d[i]
            bbox.label = int(labels_3d[i])
            arr_bbox.boxes.append(bbox)

        arr_bbox.header.frame_id = self.frame_id
        arr_bbox.header.stamp = rospy.Time.now()
        if len(arr_bbox.boxes) != 0:
            super().publish(arr_bbox)
            arr_bbox.boxes = []
        else:
            arr_bbox.boxes = []
            super().publish(arr_bbox)

```

以上代码中，调用 publish 函数来发布 bbox。

## 自定义 MarkerArray

这种方法需要使用 ros 内置的 Marker 和 MarkerArray 类，然后自己根据 bbox 的 8 个顶点自己画 bbox 的线（Marker），把这些顶点连在一起就成为框（MarkerArray），然后将这个框发布出来。代码如下：

```python
import rospy
from geometry_msgs.msg import Point
from visualization_msgs.msg import MarkerArray, Marker
# from mmdet3d.core.bbox.box_np_ops import boxes3d_to_corners3d_lidar  # 老版本
from mmdet3d.structures.ops.box_np_ops import boxes3d_to_corners3d_lidar  # 新版本


class MkArrBBPub(rospy.Publisher):
    """
    marker array bbox 3d publisher
    """

    def __init__(self, name='bbox3d', frame_id='rslidar', rate=10):
        super().__init__(name, MarkerArray, queue_size=2)  # ####
        self.name = name
        self.frame_id = frame_id
        self.rate = rate
        self.life_time = 1.0 / rate
        self.lines = [[0, 1], [1, 2], [2, 3], [3, 0],  # upper face
                      [4, 5], [5, 6], [6, 7], [7, 4],  # lower face
                      [4, 0], [5, 1], [6, 2], [7, 3],  # side face
                      [3, 4], [0, 7]]  # front face

    def _marker_common_setter(self, marker):
        marker.header.frame_id = self.frame_id
        marker.header.stamp = rospy.Time.now()
        marker.action = Marker.ADD
        marker.lifetime = rospy.Duration(self.life_time)
        self._marker_color_setter(marker, [0.0, 1.0, 1.0, 1.0])

    def _marker_color_setter(self, marker, color: list):
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]

    def _marker_scale_setter(self, marker, scale):
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale

    def _marker_orientation_setter(self, marker, orientation: list):
        marker.pose.orientation.x = orientation[0]
        marker.pose.orientation.y = orientation[1]
        marker.pose.orientation.z = orientation[2]
        marker.pose.orientation.w = orientation[3]

    def _box_marker_setter(self, box_marker, corners_3d_velo, i):
        box_marker.type = Marker.LINE_LIST
        box_marker.id = i
        self._marker_common_setter(box_marker)
        self._marker_scale_setter(box_marker, 0.1)
        self._marker_orientation_setter(box_marker, [0, 0, 0, 1])
        box_marker.points = []
        for line in self.lines:
            p1 = corners_3d_velo[line[0]]
            box_marker.points.append(Point(p1[0], p1[1], p1[2]))
            p2 = corners_3d_velo[line[1]]
            box_marker.points.append(Point(p2[0], p2[1], p2[2]))

    def _id_marker_setter(self, id_marker, corners_3d_velo, track_id, i):
        id_marker.id = i + 1000
        id_marker.type = Marker.TEXT_VIEW_FACING
        self._marker_common_setter(id_marker)
        self._marker_scale_setter(id_marker, 1)
        p4 = corners_3d_velo[4]  # upper front left corner
        id_marker.pose.position.x = p4[0]
        id_marker.pose.position.y = p4[1]
        id_marker.pose.position.z = p4[2] + 0.5
        id_marker.text = str(track_id[i])

    def publish(self, boxes_3d):
        """
        1.将结果根据置信度（score_thr）过滤
        2.发布话题
        Args:
            det_result: dict: 检测结果
                boxes_3d: Nx7 or Nx9 xyzwlhr
                scores_3d: Nx1
                labels_3d: Nx1
            score_thr: int：置信度

        Returns:

        """

        corners_3d_velos = boxes3d_to_corners3d_lidar(boxes_3d)

        marker_array = MarkerArray()
        for i, corners_3d_velo in enumerate(corners_3d_velos):
            # 3d box
            box_marker = Marker()
            self._box_marker_setter(box_marker, corners_3d_velo, i)
            marker_array.markers.append(box_marker)

        super().publish(marker_array)

```

以上代码中，同样调用 publish 函数来发布 bbox。

## 日期

2023/05/26：文章创作日期

2023/07：技术研究日期
