"""
元宇宙数据可视化
"""
import numpy as np
import open3d
import mmcv
import os.path as osp
import os
import json
import math


class Open3D_visualizer():

    def __init__(self, points, gt_bboxes, pred_bboxes) -> None:
        self.vis = open3d.visualization.Visualizer()
        self.points = self.points2o3d(points)
        self.gt_boxes = self.box2o3d(gt_bboxes,
                                     'red') if gt_bboxes is not None else None
        self.pred_boxes = self.box2o3d(
            pred_bboxes, 'green') if pred_bboxes is not None else None

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

        l = size[0] / 2
        w = size[1] / 2
        h = size[2] / 2
        R = self.rotz(1 * heading_angle)

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
    points = points.reshape(-1, 5)[:, [0, 1, 2]]
    return points


class Metadataset:

    def __init__(self, dataroot) -> None:
        self.dataroot = dataroot
        self.point_root = osp.join(dataroot, 'Lidar')
        self.anno_root = osp.join(dataroot, 'jsonall')

    def load_points(self, file_path):
        """
        读取点云文件, 返回点云和索引(哪一帧)
        return:
            pc: np.array, shape(N, 5)
        """
        index = file_path.split('.')[0].split('_')[-1]
        pc = load_pointcloud(file_path)
        return pc, int(index)

    def load_gt_boxes(self, index):
        """
        从六个相机对应的文件夹读取对应帧的 gt_boxes 和 外参矩阵，将 ego 下的 gt_boxes 转换成 lidar 下的 gt_boxes
        """

        json_filename = osp.join(self.anno_root,
                                 str(index) + '_BoundsInfo.json')
        with open(json_filename, 'r') as f:
            data = json.load(f)
        gt_bboxes = self._get_boxes(data)

        return gt_bboxes

    def _get_boxes(self, json_data):
        boxes = []
        objects = json_data['objects']
        for obj in objects:
            bboxes3d = obj['bboxes3D']
            x, y, z = bboxes3d['centerPosition']['x'], bboxes3d[
                'centerPosition']['y'], bboxes3d['centerPosition']['z']
            l, w, h = bboxes3d['lengthWidthHeight']['x'], bboxes3d[
                'lengthWidthHeight']['y'], bboxes3d['lengthWidthHeight']['z']
            yaw = bboxes3d['eulerAngles']['z']
            bbox = [x, y, z, l, w, h, yaw]
            boxes.append(bbox)
        boxes = np.array(boxes)
        return boxes

    def show(self):
        for idx, file_name in enumerate(os.listdir(self.point_root)):
            pts_filename = osp.join(self.point_root, file_name)
            points, index = self.load_points(pts_filename)
            points[:, 1] = points[:, 1]
            gt_bboxes = self.load_gt_boxes(index)
            o3dvis = Open3D_visualizer(points,
                                       gt_bboxes=gt_bboxes,
                                       pred_bboxes=None)
            o3dvis.show()


if __name__ == '__main__':

    dataroot = '/home/daxiongpro/2tb/datasets/meta_dataset/Out0703/'
    meta = Metadataset(dataroot)
    meta.show()
