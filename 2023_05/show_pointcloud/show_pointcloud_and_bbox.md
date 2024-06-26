# open3d

使用 open3d 可视化点云、预测框、真值。可视化点云比较容易。主要是画框。首先介绍如何读取点云。然后介绍画框的两种方法。

## 读取点云

直接上代码。

```python
def load_pointcloud(pts_filename):
    """
    读取点云文件
    返回 np.array, shape(N, 3)
    """
    # 加载点云
    mmcv.check_file_exist(pts_filename)
    if pts_filename.endswith('.npy'):
        points = np.load(pts_filename)
    else:
        points = np.fromfile(pts_filename, dtype=np.float32)
    # 转换点云格式
    points = points.reshape(-1, 6)[:, [0, 1, 2]]
    return points
```

## 画框方法一

为了简便，可以使用 mmdetection3d api。

```python
import numpy as np
import torch
from mmengine.structures import InstanceData
from mmdet3d.structures import (DepthInstance3DBoxes, Det3DDataSample)
from mmdet3d.visualization import Det3DLocalVisualizer


def show_data(points=np.random.rand(1000, 3), bbox=torch.rand((5, 7))):
    """show point cloud data with openmmlab and open3d.

    :param points: point clouds, defaults to np.random.rand(1000, 3)
    :type points: numpy ndarray, optional
    :param bbox: bounding boxes, (xzywlhr), defaults to torch.rand((5, 7))
    :type bbox: numpy ndarray, optional
    """
    det3d_local_visualizer = Det3DLocalVisualizer()

    # points = np.random.rand(1000, 3)

    gt_instances_3d = InstanceData()
    gt_instances_3d.bboxes_3d = DepthInstance3DBoxes(bbox)
    # gt_instances_3d.labels_3d = torch.randint(0, 2, (5, ))

    gt_det3d_data_sample = Det3DDataSample()
    gt_det3d_data_sample.gt_instances_3d = gt_instances_3d

    data_input = dict(points=points)

    det3d_local_visualizer.add_datasample(
        '3D Scene',
        data_input,
        gt_det3d_data_sample,
        vis_task='lidar_det',
        show=True)

show_data()
```

使用时，将 points 和 bbox 换成自己的数据就行。

> points 表示点云数据，数据格式为np.ndarray，结构为(N, d)。N 表示点的数量。D 表示维度，D >=3。当 D>3 时，会自动处理成 (N, 3)；
>
> bbox 表示包围框，数据格式为 np.ndarray，结构为(N, 7)。N 表示包围框的数量。7 表示包围框的 7 个回归值，分别为：x,y,z,l,w,h,r。其中，xyz 表示包围框底边中心点的坐标，lwh 表示长宽高，r 表示包围框绕 z 轴的旋转角 yaw。

运行代码，可视化结果如下：

![pic1](./image/pic1.png "pic1")

使用自己的数据集中的 points 和 bbox，可视化结果如下：

![pic2](./image/pic2.png "pic2")

## 画框方法二

使用 open3d 手工绘制 bbox 的线条。

```python

import numpy as np
import open3d
import mmcv


class Open3D_visualizer():

    def __init__(self, points, gt_bboxes, pred_bboxes) -> None:
        self.vis = open3d.visualization.Visualizer()
        self.points = self.points2o3d(points)
        self.gt_boxes = self.box2o3d(gt_bboxes, 'red') if gt_bboxes is not None else None
        self.pred_boxes = self.box2o3d(pred_bboxes, 'green') if pred_bboxes is not None else None

    def points2o3d(self, points):
        """
        points: np.array, shape(N, 3)
        """
        pointcloud = open3d.geometry.PointCloud()
        pointcloud.points = open3d.utility.Vector3dVector(points)
        pointcloud.colors = open3d.utility.Vector3dVector(
            [[255, 255, 255] for _ in range(len(points))])
        return pointcloud

    def box2o3d(self, bboxes, color):
        """
        bboxes: np.array, shape(N, 7)
        color: 'red' or 'green'
        """

        bbox_lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7],
                      [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
        if color == 'red':
            colors = [[1, 0, 0] for _ in range(len(bbox_lines))]  # red
        elif color == 'green':
            colors = [[0, 1, 0] for _ in range(len(bbox_lines))]  # green
        else:
            print("请输入 green 或者 red。green 表示预测框，red 表示真值框。")

        all_bboxes = open3d.geometry.LineSet()
        for bbox in bboxes:  
            corners_3d = self.compute_box_3d(bbox[0:3], bbox[3:6], bbox[6])
            o3d_bbox = open3d.geometry.LineSet()
            o3d_bbox.lines = open3d.utility.Vector2iVector(bbox_lines)
            o3d_bbox.colors = open3d.utility.Vector3dVector(colors)
            o3d_bbox.points = open3d.utility.Vector3dVector(corners_3d)
            all_bboxes += o3d_bbox

        return all_bboxes

    def compute_box_3d(self, center, size, heading_angle):
        """
        计算 box 的 8 个顶点坐标
        """
        h = size[2]
        w = size[0]
        l = size[1]
        heading_angle = -heading_angle - np.pi / 2

        center[2] = center[2] + h / 2
        R = self.rotz(1 * heading_angle)
        l = l / 2
        w = w / 2
        h = h / 2
        x_corners = [-l, l, l, -l, -l, l, l, -l]
        y_corners = [w, w, -w, -w, w, w, -w, -w]
        z_corners = [h, h, h, h, -h, -h, -h, -h]
        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        corners_3d[0, :] = center[0] + corners_3d[0, :]
        corners_3d[1, :] = center[1] + corners_3d[1, :]
        corners_3d[2, :] = center[2] + corners_3d[2, :]
        return np.transpose(corners_3d)

    def rotz(self, t):
        """Rotation about the z-axis."""
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def show(self):
        # 创建窗口
        self.vis.create_window(window_name="Open3D_visualizer")
        opt = self.vis.get_render_option()
        opt.point_size = 1
        opt.background_color = np.asarray([0, 0, 0])
        # 添加点云、真值框、预测框
        self.vis.add_geometry(self.points)
        if self.gt_boxes is not None:
            self.vis.add_geometry(self.gt_boxes)
        if self.pred_boxes is not None:
            self.vis.add_geometry(self.pred_boxes)

        self.vis.get_view_control().rotate(180.0, 0.0)
        self.vis.run()


def load_pointcloud(pts_filename):
    """
    读取点云文件
    返回 np.array, shape(N, 3)
    """
    # 加载点云
    mmcv.check_file_exist(pts_filename)
    if pts_filename.endswith('.npy'):
        points = np.load(pts_filename)
    else:
        points = np.fromfile(pts_filename, dtype=np.float32)
    # 转换点云格式
    points = points.reshape(-1, 6)[:, [0, 1, 2]]
    return points


if __name__ == '__main__':
    index = 4
    pts_filename = f'/path/to/your/point/cloud/file.bin'
    gt_filename = f'/path/to/your/gt/file.pkl'
    pred_filename = f'/path/to/your/pred/file.pkl'

    points = load_pointcloud(pts_filename)
    #  使用 mmcv.load 读取真值和预测框的 pkl，获取对应的 bboxes。bboxes 格式为 np.array，shape 为 (N, 3)
    gt_bboxes = ...
    pred_bboxes = ...
    o3dvis = Open3D_visualizer(points, gt_bboxes, pred_bboxes)
    o3dvis.show()

```

修改点云、bbox 的路径参数，读取点云、bbox。可视化结果如下：

![pic3](./image/pic3.png "pic3")

## 日期

2024/04/15：新增读取点云数据、手工绘制 bbox(画框方法二)

2023/05/11：更新

* xyz 表示 bbox 底边中点
* 注释代码中的 label
* 增加数据集配图

2023/05/10：创作日期
