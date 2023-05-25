# 使用 mmdet3d 框架训练自定义数据集

mmdet3d 框架是商汤的 3D 目标检测工具箱，支持训练、评测、可视化。详情：[mmdet3d 官网](https://github.com/open-mmlab/mmdetection3d) 、[官方文档](https://mmdetection3d.readthedocs.io/zh_CN/latest/tutorials/customize_dataset.html)。官方文档中，提供了 3 种不同的方式：支持新的数据格式、将新数据的格式转换为现有数据的格式、将新数据集的格式转换为一种当前可支持的中间格式。这三种方式大同小异。本质都是 4 步：

1. 将数据集预处理成 pkl 文件
2. 创建自定义数据集类，并在数据集中读取并解析 pkl 文件
3. 创建训练的配置文件
4. 创建评价类 Metric

其中，第 4 步在训练的过程中可有可无，在验证和测试的时候需要。

为了简便，本文只使用激光雷达数据作为例子，自定义数据集的名称叫 "meg"。使用的模型为 [3DSSD](https://github.com/open-mmlab/mmdetection3d/tree/main/configs/3dssd) 。

## 可视化数据集

可视化数据集的目的是看一下数据集的内容是否正确，GT bbox 是否能够准确框出场景中的障碍物信息。方法参考 [open3d 显示 pointcloud 和 bbox](2023_05/show_pointcloud/show_pointcloud_and_bbox.md) 。

## 数据集预处理成 pkl 文件

mmdet3d 官方提供的数据集的 annotation infomation （以下简称 ann info）都会先预处理成 pkl 文件，文件内保存一些数据集的基本信息，如数据存放路径，数据真值(bboxes、labels)， 相机内外参等。通常，自定义的数据集的 ann info 保存格式可能有很多，如 json、yaml 等。我们需要将自定义的数据集的 ann info 保存成 pkl 文件。这里可以模仿 nuscenes 数据集的创建过程，步骤如下：

### 主函数

```python
# create_data.py 
import argparse
from projects.Meg_Dataset.meg_dataset import meg_converter as meg_converter


def meg_data_prep(root_path,
                  info_prefix,
                  dataset_name,
                  out_dir):

    meg_converter.create_meg_infos(root_path, info_prefix)

parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('dataset', metavar='kitti', help='name of the dataset')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/kitti',
    help='specify the root path of dataset')
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/kitti',
    required=False,
    help='name of info pkl')
parser.add_argument('--extra-tag', type=str, default='kitti')
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
args = parser.parse_args()

if __name__ == '__main__':
    from mmdet3d.utils import register_all_modules
    register_all_modules()

    if args.dataset == 'meg':
        meg_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            dataset_name='MegDataset',
            out_dir=args.out_dir)
    else:
        raise NotImplementedError(f'Don\'t support {args.dataset} dataset.')

```

上述代码中，主函数先读取参数，然后最关键的就是这个函数：`meg_converter.create_meg_infos` 。

### 整理并导出成 pkl 文件

`create_meg_info` 首先解读数据集，整理成 python 的字典格式，其包括两个值：`data_list` 和 `meta_info` 。然后再使用 `mmengine.dump` 保存成 pkl 文件，它的定义以及实现如下：

```python
# meg_converter.py
import os
from os import path as osp
import mmengine
from pyquaternion import Quaternion
import json

class_names = [
    "小汽车", "汽车", "货车", "工程车", "巴士", "摩托车", "自行车", "三轮车", "骑车人", "骑行的人", "人",
    "行人", "其它", "残影", "蒙版", "其他", "拖挂", "锥桶", "防撞柱"
]


def create_meg_infos(root_path, info_prefix):
    """
    处理数据集的激光雷达数据
    """

    metainfo = dict(class_names=class_names)

    train_infos, val_infos = _fill_trainval_infos(root_path)

    if train_infos is not None:
        data = dict(data_list=train_infos, metainfo=metainfo)
        info_path = osp.join(root_path,
                             '{}_infos_train.pkl'.format(info_prefix))
        mmengine.dump(data, info_path)

    if val_infos is not None:
        data['data_list'] = val_infos
        info_val_path = osp.join(root_path,
                                 '{}_infos_val.pkl'.format(info_prefix))
        mmengine.dump(data, info_val_path)


def _fill_trainval_infos(root_path):
    """
    划分训练集和验证集
    """

    train_infos = []
    val_infos = []
  
    """
    这部分自己写，步骤：
    1. 读取自定义数据集的 ann info 文件
    2. 将自定义的 ann info 信息处理成 dict
    3. 将 dict 存入 train_infos 和 val_infos 列表
    """

    return train_infos, val_infos

```

上述代码中，`class_names` 更换为自定义数据集的类别。然后编写 `_fill_trainval_infos(root_path)` 函数，方法见函数内注释。运行 `create_data.py` 文件后，就会在数据集根目录下保存 pkl 文件。

### 查看处理好的 pkl 文件

在处理好 ann info 的 pkl 文件后，可以读取 pkl 文件查看。读取方法：

```python
import pickle

# 读取 pickle 文件
p = 'data/nuscenes-mini/nuscenes_infos_train.pkl'
with open(p, 'rb') as f:
    data = pickle.load(f)

# 打印数据
print(data)

```

可以在 `print(data)` 打一个断点，用 vscode 或 pycharm 的 debug 模式去查看。没什么问题就可以进行下一步：创建自定义数据集类。

## 创建自定义数据集类

上面处理完的 pkl 文件只是保存一些基本信息，如数据集文件路径、ann info 的一些基本格式（如列表、字典等）。自定义数据集类的作用是从路径中读取数据 ann info，并处理成 mmdet3d 的格式。

回顾一下，使用 pytorch 训练自定义数据集时，也需要创建自定义数据集类，其中作主要的就是编写 `__getitem__`和 `__len__` 这两个函数，其作用是读取数据和 GT 信息，解析成 numpy 或者 Tensor 格式并 return。mmdet3d 中，也要进行类似的操作，但是因为读取数据部分放在 pipeline 中，所以只需要解析 ann info 并 return 即可。mmdet3d 的包围框 gt_bboxes_3d 有固定的格式：LiDARInstance3DBoxes 。创建数据集的代码如下：

```python
from typing import Callable, List, Union

import numpy as np

from mmdet3d.registry import DATASETS
from mmdet3d.structures import LiDARInstance3DBoxes
from mmdet3d.datasets.det3d_dataset import Det3DDataset


@DATASETS.register_module()
class MegDataset(Det3DDataset):

    METAINFO = {
        'classes':
        ("小汽车", "汽车", "货车", "工程车", "巴士", "摩托车", "自行车", "三轮车", "骑车人", "骑行的人",
         "人", "行人", "其它", "残影", "蒙版", "其他", "拖挂", "锥桶", "防撞柱")
    }

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 pipeline: List[Union[dict, Callable]] = [],
                 **kwargs) -> None:

        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            **kwargs)

    def parse_data_info(self, info: dict) -> dict:
        """Process the raw data info.

        Convert all relative path of needed modality data file to
        the absolute path. And process the `instances` field to
        `ann_info` in training stage.

        Args:
            info (dict): Raw info dict.

        Returns:
            dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """

        if not self.test_mode:
            # used in training
            info['ann_info'] = self.parse_ann_info(info)
        if self.test_mode and self.load_eval_anns:
            info['eval_ann_info'] = self.parse_ann_info(info)

        return info

    def parse_ann_info(self, info: dict) -> dict:
        """Process the `instances` in data info to `ann_info`.

        Args:
            info (dict): Data information of single data sample.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                  3D ground truth bboxes.
                - gt_labels_3d (np.ndarray): Labels of ground truths.
        """
        ann_info = super().parse_ann_info(info)

        if ann_info is None:
            # empty instance
            ann_info = dict()
            ann_info['gt_bboxes_3d'] = np.zeros((0, 7), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros(0, dtype=np.int64)

        gt_bboxes_3d = LiDARInstance3DBoxes(
            ann_info['gt_bboxes_3d'],
            box_dim=ann_info['gt_bboxes_3d'].shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        ann_info['gt_bboxes_3d'] = gt_bboxes_3d

        return ann_info

```

训练的时候，首先进入 `parse_data_info` 函数，然后调用 `parse_ann_info` 函数。其中的参数 `info` 是 读取的 pkl 格式。其中，METAINFO 根据自定义数据集的实际类别情况修改。下面 gt_bboxes_3d 也根据实际情况自行编写：

```python
gt_bboxes_3d = LiDARInstance3DBoxes(
    ann_info['gt_bboxes_3d'],
    box_dim=ann_info['gt_bboxes_3d'].shape[-1],
    origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)
```

## 编写配置文件

本文以 3DSSD 模型为例子。mmdet3d 框架的 config 文件采用继承的方式，详情见[官方文档](https://mmdetection3d.readthedocs.io/zh_CN/latest/tutorials/config.html)。但是笔者个人建议对继承这成方式不熟悉的用户不使用这种方式，而是将配置文件写全。可以先运行一遍官方的 3dssd 的 [config](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/3dssd/3dssd_4xb4_kitti-3d-car.py)，然后在 work_dirs 找到其完整的配置文件，复制到创建的配置文件。在创建的配置文件顶部，可以自定义导入某些 python 模块。

```python
# 3dssd_meg-19classes.py
custom_imports = dict(
    imports=[
        'projects.Meg_Dataset.meg_dataset.meg_dataset',
        'projects.Meg_Dataset.meg_dataset.loading'
    ],
    allow_failed_imports=False)
```

接着再进行修改，修改的部分主要有：数据集名称、data_root、ann_file 路径、class_names，num_classes(model->bbox_head->num_classes)。其中值得注意的是，num_classes 需要对应 class_names 列表的长度，笔者在训练时，在这上面花费了很长时间 debug 都没找到问题所在，最终还是靠度娘。

然后再改一些超参：比如 lr、batch_size 、max_epochs、val_interval 等。val_interval 表示每训练多少个 epoch 验证一次结果。

mmdet3d 官方在训练的时候会每过几个 epoch 进行一次 val 验证，但是验证需要编写 Metric 类，这部分笔者还没仔细研究，因此笔者训练的时候不进行验证，把一切关于 val 的内容注释，否则会报错。完整的代码如下：

```python
# 3dssd_meg-19classes.py
custom_imports = dict(
    imports=[
        'projects.Meg_Dataset.meg_dataset.meg_dataset',
        'projects.Meg_Dataset.meg_dataset.loading'
    ],
    allow_failed_imports=False)

dataset_type = 'MegDataset'
data_root = 'data/xxxx/'
ann_file = 'meg_infos_train.pkl'
launcher = 'none'
work_dir = './work_dirs/3dssd_meg-3d-19classes'
class_names = [
    '小汽车', '汽车', '货车', '工程车', '巴士', '摩托车', '自行车', '三轮车', '骑车人', '骑行的人', '人',
    '行人', '其它', '残影', '蒙版', '其他', '拖挂', '锥桶', '防撞柱'
]

model = dict(
    type='SSD3DNet',
    data_preprocessor=dict(type='Det3DDataPreprocessor'),
    backbone=dict(
        type='PointNet2SAMSG',
        in_channels=3,
        num_points=(4096, 512, (256, 256)),
        radii=((0.2, 0.4, 0.8), (0.4, 0.8, 1.6), (1.6, 3.2, 4.8)),
        num_samples=((32, 32, 64), (32, 32, 64), (32, 32, 32)),
        sa_channels=(((16, 16, 32), (16, 16, 32), (32, 32, 64)),
                     ((64, 64, 128), (64, 64, 128), (64, 96, 128)),
                     ((128, 128, 256), (128, 192, 256), (128, 256, 256))),
        aggregation_channels=(64, 128, 256),
        fps_mods=('D-FPS', 'FS', ('F-FPS', 'D-FPS')),
        fps_sample_range_lists=(-1, -1, (512, -1)),
        norm_cfg=dict(type='BN2d', eps=0.001, momentum=0.1),
        sa_cfg=dict(
            type='PointSAModuleMSG',
            pool_mod='max',
            use_xyz=True,
            normalize_xyz=False)),
    bbox_head=dict(
        type='SSD3DHead',
        vote_module_cfg=dict(
            in_channels=256,
            num_points=256,
            gt_per_seed=1,
            conv_channels=(128, ),
            conv_cfg=dict(type='Conv1d'),
            norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.1),
            with_res_feat=False,
            vote_xyz_range=(3.0, 3.0, 2.0)),
        vote_aggregation_cfg=dict(
            type='PointSAModuleMSG',
            num_point=256,
            radii=(4.8, 6.4),
            sample_nums=(16, 32),
            mlp_channels=((256, 256, 256, 512), (256, 256, 512, 1024)),
            norm_cfg=dict(type='BN2d', eps=0.001, momentum=0.1),
            use_xyz=True,
            normalize_xyz=False,
            bias=True),
        pred_layer_cfg=dict(
            in_channels=1536,
            shared_conv_channels=(512, 128),
            cls_conv_channels=(128, ),
            reg_conv_channels=(128, ),
            conv_cfg=dict(type='Conv1d'),
            norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.1),
            bias=True),
        objectness_loss=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        center_loss=dict(
            type='mmdet.SmoothL1Loss', reduction='sum', loss_weight=1.0),
        dir_class_loss=dict(
            type='mmdet.CrossEntropyLoss', reduction='sum', loss_weight=1.0),
        dir_res_loss=dict(
            type='mmdet.SmoothL1Loss', reduction='sum', loss_weight=1.0),
        size_res_loss=dict(
            type='mmdet.SmoothL1Loss', reduction='sum', loss_weight=1.0),
        corner_loss=dict(
            type='mmdet.SmoothL1Loss', reduction='sum', loss_weight=1.0),
        vote_loss=dict(
            type='mmdet.SmoothL1Loss', reduction='sum', loss_weight=1.0),
        num_classes=len(class_names),
        bbox_coder=dict(
            type='AnchorFreeBBoxCoder', num_dir_bins=12, with_rot=True)),
    train_cfg=dict(
        sample_mode='spec', pos_distance_thr=10.0, expand_dims_length=0.05),
    test_cfg=dict(
        nms_cfg=dict(type='nms', iou_thr=0.1),
        sample_mode='spec',
        score_thr=0.0,
        per_class_proposal=True,
        max_output_num=100))

point_cloud_range = [0, -40, -5, 70, 40, 3]
input_modality = dict(use_lidar=True, use_camera=False)
metainfo = dict(class_names=class_names)
db_sampler = dict()
train_pipeline = [
    dict(type='MegLoadPointsFromFile', coord_type='LIDAR', use_dim=3),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='PointsRangeFilter', point_cloud_range=[0, -40, -5, 70, 40, 3]),
    dict(type='ObjectRangeFilter', point_cloud_range=[0, -40, -5, 70, 40, 3]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='ObjectNoise',
        num_try=100,
        translation_std=[1.0, 1.0, 0],
        global_rot_range=[0.0, 0.0],
        rot_range=[-1.0471975511965976, 1.0471975511965976]),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.9, 1.1]),
    dict(type='PointSample', num_points=16384),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='MegLoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=3,
        use_dim=3),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            # dict(
            #     type='GlobalRotScaleTrans',
            #     rot_range=[0, 0],
            #     scale_ratio_range=[1.0, 1.0],
            #     translation_std=[0, 0, 0]),
            # dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter',
                point_cloud_range=[0, -40, -5, 70, 40, 3]),
            dict(type='PointSample', num_points=16384)
        ]),
    dict(type='Pack3DDetInputs', keys=['points'])
]
eval_pipeline = [
    dict(
        type='MegLoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=3,
        use_dim=3),
    dict(type='Pack3DDetInputs', keys=['points'])
]
train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=ann_file,
            data_prefix=dict(pts='training/velodyne_reduced'),
            pipeline=train_pipeline,
            modality=dict(use_lidar=True, use_camera=False),
            test_mode=False,
            metainfo=dict(class_names=class_names),
            box_type_3d='LiDAR')))
# val_dataloader = dict(
#     batch_size=1,
#     num_workers=1,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         data_prefix=dict(pts='training/velodyne_reduced'),
#         ann_file=ann_file,
#         pipeline=eval_pipeline,
#         modality=dict(use_lidar=True, use_camera=False),
#         test_mode=True,
#         metainfo=dict(class_names=class_names),
#         box_type_3d='LiDAR'))
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts='training/velodyne_reduced'),
        ann_file=ann_file,
        pipeline=test_pipeline,
        modality=dict(use_lidar=True, use_camera=False),
        test_mode=True,
        metainfo=dict(class_names=class_names),
        box_type_3d='LiDAR'))
# val_evaluator = dict(
#     type='KittiMetric',
#     ann_file=data_root + ann_file,
#     metric='bbox')
test_evaluator = dict(
    type='KittiMetric', ann_file=data_root + ann_file, metric='bbox')
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
default_scope = 'mmdet3d'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False
file_client_args = dict(backend='disk')
lr = 0.002
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.0),
    clip_grad=dict(max_norm=35, norm_type=2))
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=180, val_interval=-1)
# val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=80,
        by_epoch=True,
        milestones=[45, 60],
        gamma=0.1)
]

```

官方代码的 train_pipeline 中，使用的是 LoadPointsFromFile 类，但需要适配自定义数据集的 pkl 格式，可能需要自定义 LoadPointsFromFile 类。上述代码的 train_pipeline 中，使用了自定义的 MegLoadPointsFromFile 类。MegLoadPointsFromFile 代码如下：

```python
import numpy as np
from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures.points import get_points_type
from mmdet3d.datasets.transforms.loading import LoadPointsFromFile
# from pyntcloud import PyntCloud


@TRANSFORMS.register_module()
class MegLoadPointsFromFile(LoadPointsFromFile):

    def _load_pcd_points(self, pts_filename: str) -> np.ndarray:
        """读取 pcd 文件， 得到 np.ndarray(N, 4)
        """
        with open(pts_filename, 'rb') as f:
            data = f.read()
            data_binary = data[data.find(b"DATA binary") + 12:]
            points = np.frombuffer(data_binary, dtype=np.float32).reshape(-1, 3)
            points = points.astype(np.float32)
        return points

    def transform(self, results: dict) -> dict:
        """
            生成 LiDARPoints 格式
        """
        pts_file_path = results['lidar_path']
        points = self._load_pcd_points(pts_file_path)  # (N, 4)

        points = points[:, self.use_dim]

        points_class = get_points_type(self.coord_type)
        points = points_class(points, points_dim=points.shape[-1])
        results['points'] = points

        return results

```

上述代码中，程序首先进入 `transform` 函数，然后调用  `_load_pcd_points` 函数，`_load_pcd_points` 函数的实现过程需要自行替换。有的数据集点云文件是 pcd 格式，有的是 bin 格式保存，需要根据不同的格式读取，并解析成 numpy 格式。

## 开始训练

在完成上述所有的代码后，即可进行训练：

```bash
python tools/train.py xxx/3dssd_meg-19classes.py
```

不出意外的话，可以训练：

```bash
05/25 13:45:47 - mmengine - INFO - ------------------------------
05/25 13:45:47 - mmengine - INFO - The length of the dataset: 685
05/25 13:45:47 - mmengine - INFO - The number of instances per category in the dataset:
+----------+--------+
| category | number |
+----------+--------+
| 小汽车   | 0      |
| 汽车     | 4285   |
| 货车     | 526    |
| 工程车   | 46     |
| 巴士     | 120    |
| 摩托车   | 324    |
| 自行车   | 178    |
| 三轮车   | 205    |
| 骑车人   | 487    |
| 骑行的人 | 0      |
| 人       | 513    |
| 行人     | 0      |
| 其它     | 15     |
| 残影     | 2      |
| 蒙版     | 695    |
| 其他     | 0      |
| 拖挂     | 0      |
| 锥桶     | 0      |
| 防撞柱   | 0      |
+----------+--------+
05/25 13:45:52 - mmengine - INFO - Auto resumed from the latest checkpoint /home/daxiongpro/code/mmdetection3d/work_dirs/3dssd_megvii-3d-car/epoch_81.pth.
Loads checkpoint by local backend from path: /home/daxiongpro/code/mmdetection3d/work_dirs/3dssd_megvii-3d-car/epoch_81.pth
05/25 13:45:52 - mmengine - INFO - Load checkpoint from /home/daxiongpro/code/mmdetection3d/work_dirs/3dssd_megvii-3d-car/epoch_81.pth
05/25 13:45:52 - mmengine - INFO - resumed epoch: 81, iter: 6966
05/25 13:45:53 - mmengine - INFO - Checkpoints will be saved to /home/daxiongpro/code/mmdetection3d/work_dirs/3dssd_megvii-3d-car.
/home/daxiongpro/code/mmdetection3d/mmdet3d/models/task_modules/coders/partial_bin_based_bbox_coder.py:220: UserWarning: __floordiv__ is deprecated, and its behavior will change in a future version of pytorch. It currently rounds toward 0 (like the 'trunc' function NOT 'floor'). This results in incorrect rounding for negative values. To keep the current behavior, use torch.div(a, b, rounding_mode='trunc'), or for actual floor division, use torch.div(a, b, rounding_mode='floor').
  angle_cls = shifted_angle // angle_per_class
05/25 13:46:44 - mmengine - INFO - Exp name: 3dssd_megvii-3d-car_20230525_134521
05/25 13:46:54 - mmengine - INFO - Epoch(train)  [82][50/86]  lr: 2.0000e-05  eta: 2:53:33  time: 1.2303  data_time: 0.6377  memory: 18906  grad_norm: 71.2818  loss: 24.2896  centerness_loss: 0.0049  center_loss: 0.7276  dir_class_loss: 1.4938  dir_res_loss: 0.0230  size_res_loss: 1.0054  corner_loss: 13.0545  vote_loss: 7.9803
```

> 3DSSD 是一个检测车辆的模型，只有 car 一个类别，而本文的 meg 数据集有 19 个类别，所以只是做一个简单的训练教程，训练出来的效果很差，loss 也很大。要想得到好的训练效果，应该选一个好的模型，并且修改模型的相关参数，例如数据集的均值、方差等等。

## 日期

2023/05/25：创作日期
