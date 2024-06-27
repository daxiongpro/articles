"""
元宇宙第一批数据可视化
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
    points = points.reshape(-1, 5)[:, [0, 1, 2]]
    return points


def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    将 roll, pitch, yaw 转换为旋转矩阵。

    参数:
    roll (float): 绕 x 轴的旋转角度（弧度）
    pitch (float): 绕 y 轴的旋转角度（弧度）
    yaw (float): 绕 z 轴的旋转角度（弧度）

    返回:
    numpy.ndarray: 形状为 (3, 3) 的旋转矩阵
    """
    sx = np.sin(roll)
    cx = np.cos(roll)
    sy = np.sin(pitch)
    cy = np.cos(pitch)
    sz = np.sin(yaw)
    cz = np.cos(yaw)

    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def transform_bbox(bboxes, R, t):
    """
    将 bbox 从一种坐标系转换到另一种坐标系。

    参数:
    bboxes (numpy.ndarray): 形状为 (n, 7) 的数组，每行代表一个 bbox，包含 [x, y, z, l, w, h, yaw]。
    R (numpy.ndarray): 形状为 (3, 3) 的旋转矩阵。
    t (numpy.ndarray): 形状为 (3,) 的位移向量。

    返回:
    numpy.ndarray: 形状为 (n, 7) 的转换后的 bbox 数组。
    """
    # 分离位置和其他参数
    positions = bboxes[:, :3]  # 形状为 (n, 3)
    sizes_and_yaw = bboxes[:, 3:]  # 形状为 (n, 4)

    # 对位置进行旋转和平移
    transformed_positions = (R @ positions.T).T + t

    # 对 yaw 进行旋转
    transformed_yaw = np.arctan2(np.sin(bboxes[:, 6]), np.cos(
        bboxes[:, 6])) + np.arctan2(np.sin(np.arctan2(R[1, 0], R[0, 0])),
                                    np.cos(np.arctan2(R[1, 0], R[0, 0])))

    # 合并位置和其他参数
    transformed_bboxes = np.hstack(
        (transformed_positions, sizes_and_yaw[:, :3],
         transformed_yaw[:, np.newaxis]))

    return transformed_bboxes


class Metadataset:
    def __init__(self, dataroot) -> None:
        self.dataroot = dataroot
        self.point_root = osp.join(dataroot, 'Lidar')
        self.anno_root = osp.join(dataroot, '2024.06.13-16.18.42/JsonData')
        
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
        gt_bboxes = []
        for dir_name in os.listdir(self.anno_root):  # 每个相机
            json_filename = osp.join(self.anno_root, dir_name, str(index) + '_BoundsInfo.json')
            with open(json_filename, 'r') as f:
                data = json.load(f)
            boxes = self._get_boxes(data)
            #  旋转平移
            matrix = self._get_ego2lidar(data)
            boxes = transform_bbox(boxes, matrix[0:3, 0:3], matrix[0:3, 3])

            gt_bboxes.append(boxes)
        gt_bboxes = np.vstack(gt_bboxes)  # xyzlhw,yaw
        return gt_bboxes
    
    def _get_boxes(self, json_data):
        boxes = []
        objects = json_data['objects']
        for obj in objects:
            bboxes3d = obj['bboxes3D']
            x, y, z = bboxes3d['centerPosition']['x'], bboxes3d['centerPosition'][
                'y'], bboxes3d['centerPosition']['z']
            l, w, h = bboxes3d['lengthWidthHeight']['x'], bboxes3d[
                'lengthWidthHeight']['y'], bboxes3d['lengthWidthHeight']['z']
            yaw = -bboxes3d['eulerAngles']['z']
            yaw = math.radians(yaw)  # 角度转弧度
            bbox = [x, y, z, l, w, h, yaw]
            boxes.append(bbox)
        boxes = np.array(boxes)
        boxes[:, 3:6] = boxes[:, 3:6] / 50  # 单位转换 cm -> m
        boxes[:, :3] = boxes[:, :3] / 100  # 单位转换 cm -> m
        return boxes
    
    def _get_ego2lidar(self, json_data):
        lidar2ego_pos = np.array([
            json_data["lidarLocationInEgo"]["x"],
            json_data["lidarLocationInEgo"]["y"],
            json_data["lidarLocationInEgo"]["z"]
        ], np.float32) / 100
        lidar2ego_rot = np.array([
            json_data["lidarRotatorInEgo"]["roll"],
            json_data["lidarRotatorInEgo"]["pitch"],
            json_data["lidarRotatorInEgo"]["yaw"]
        ], np.float32) / 180 * np.pi
        lidar2ego = np.eye(4)

        lidar2ego[0:3, 0:3] = euler_to_rotation_matrix(lidar2ego_rot[0], lidar2ego_rot[1], lidar2ego_rot[2])
        lidar2ego[0:3, 3] = np.array([lidar2ego_pos[0], lidar2ego_pos[1], lidar2ego_pos[2]])
        # print(lidar2ego)
        ego2lidar = np.linalg.inv(lidar2ego)

        return ego2lidar
    
    def show(self):
        for idx, file_name in enumerate(os.listdir(self.point_root)):
            pts_filename = osp.join(self.point_root, file_name)
            points, index = self.load_points(pts_filename)
            gt_bboxes = self.load_gt_boxes(index)
            o3dvis = Open3D_visualizer(points, gt_bboxes=gt_bboxes, pred_bboxes=None)
            o3dvis.show()
            

if __name__ == '__main__':

    dataroot = '/home/daxiongpro/2tb/datasets/meta_dataset/Out/'
    meta = Metadataset(dataroot)
    meta.show()
