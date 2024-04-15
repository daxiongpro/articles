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
    gt_instances_3d.labels_3d = torch.randint(0, 2, (5, ))

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