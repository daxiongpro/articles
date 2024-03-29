# bevfusion 训练自定义数据集

2024/01/05 更新：之前的文章大段代码看起来纷繁复杂，且对应到自己的项目中，代码必然需要更换。故删除部分代码，并添加相关注释，使得文章整体脉络更清楚。

bevfusion 是一个融合了图像和雷达的 3D 目标检测网络，在当时 nuscenes 数据集上达到了 SOTA，其[官方 github 点击此处](https://github.com/mit-han-lab/bevfusion)。本文介绍如如何使用自定义数据集 meg，来训练 bevfusion 。之前的文章“[使用 mmdet3d 框架训练自定义数据集](https://zhuanlan.zhihu.com/p/632104137)"中，介绍了如何使用 mmdetection3d，训练自定义数据集。由于 bevfusion 也是基于 mmdetection3d 这个仓库，因此，可以参考这篇文章来处理数据，只不过在数据预处理和加载部分，要同时处理点云和图像两种模态的数据。

在做项目的时候，建议不要在原作者源码上修改，而要充分利用面向对象的继承、多态的特性，新建源代码文件，以包导入的方式进行编程。在这里，我们在项目根目录(bevfusion)下新建 projects 文件夹，然后在这个文件夹下新建项目。项目目录如下：

```
projects
└── Meg_dataset
    ├── bash_runner
    │   ├── train.sh
    │   ├── create_data.sh
    │   ├── env_install.sh
    │   ├── test.sh
    │   └── vis.sh
    ├── configs
    │   ├── _base_
    │   │   └── default_runtime.py
    │   └── bevfusion_c_l_meg.py
    ├── mmdet3d
    │   ├── core
    │   │   └── visualize.py
    │   ├── datasets
    │   │   └── meg_dataset.py
    │   ├── evaluate
    │   │   └── map.py
    │   └── models
    │       ├── bevfusion_simple.py
    │       ├── centerhead_without_vel.py
    │       ├── lss_transform_simple.py
    │       └── pillar_encoder.py
    ├── README.md
    └── tools
        ├── create_data.py
        ├── data_convert
        │   ├── create_gt_database.py
        │   └── meg_converter.py
        ├── test.py
        ├── train.py
        └── visualize.py
```

> 在projects文件夹中，我们新建项目 Meg_dataset，然后仿照 mmdetection3d 的目录结构进行构造。

## 1.数据集预处理成 pkl 文件

具体步骤同前文“[使用 mmdet3d 框架训练自定义数据集](https://zhuanlan.zhihu.com/p/632104137)"。

### 1.1.主函数

主函数是数据集预处理的入口函数。

```python
# projects/tools/create_data.py
import argparse
import data_convert.meg_converter as meg_converter
from data_convert.create_gt_database import create_groundtruth_database


def meg_data_prep(root_path, info_prefix, dataset_name, out_dir):
    meg_converter.create_meg_infos(root_path, info_prefix)
    # create_groundtruth_database(dataset_name,
    #                             root_path,
    #                             info_prefix,
    #                             f"{out_dir}/{info_prefix}_infos_train.pkl")


parser = argparse.ArgumentParser(description="Data converter arg parser")
parser.add_argument("dataset", metavar="MegDataset", help="name of the dataset")
parser.add_argument(
    "--root-path",
    type=str,
    default="/",
    help="specify the root path of dataset",
)
parser.add_argument(
    "--out-dir",
    type=str,
    default="/",
    required=False,
    help="name of info pkl",
)
parser.add_argument("--extra-tag", type=str, default="meg")


args = parser.parse_args()

if __name__ == "__main__":

    if args.dataset == "meg":

        meg_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            dataset_name="MegDataset",
            out_dir=args.out_dir
        )

```

> 上述代码中，注释了 create_groundtruth_database 函数，原因是在此项目中不需要点云的真值来给点云数据增强。但是这部分代码也做了，见下文“保存点云真值"。

### 1.2.处理 annos 数据

annos 数据包括激光真值、图像真值、旋转矩阵、相机内外参等。

```python
# projects/tools/data_convert/meg_converter.py
import os
from os import path as osp
import mmcv
import numpy as np
import json
from pyquaternion import Quaternion
from projects.Meg_dataset.mmdet3d.datasets.meg_dataset import MegDataset
import math


def get_train_val_scenes(root_path):
    # 划分训练集和测试集...
    return train_scenes, val_scenes  # ['0', '1', '2', ...]


def create_meg_infos(
    root_path, info_prefix
):
    train_scenes, val_scenes = get_train_val_scenes(root_path)

    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(root_path, train_scenes, val_scenes)

    metadata = dict(version="v1.0-mini")

    print(
        "train sample: {}, val sample: {}".format(
            len(train_nusc_infos), len(val_nusc_infos)
        )
    )
    data = dict(infos=train_nusc_infos, metadata=metadata)
    info_path = osp.join(root_path, "{}_infos_train.pkl".format(info_prefix))
    mmcv.dump(data, info_path)
    data["infos"] = val_nusc_infos
    info_val_path = osp.join(root_path, "{}_infos_val.pkl".format(info_prefix))
    mmcv.dump(data, info_val_path)


def _fill_trainval_infos(root_path, train_scenes, val_scenes, test=False):

    train_nusc_infos = []
    val_nusc_infos = []

    available_scene_names = train_scenes + val_scenes

    for sid, scenes_json in enumerate(available_scene_names):
        for 每帧数据:
            if sample['is_key_frame']:
                # 获取数据集的 info...
                # dataset infos
                info = {
                    "frame_id": frame_id,
                    "lidar_path": lidar_path,
                    "sweeps": [],
                    "cams": dict(),
                    "lidar2ego_translation": lidar2ego_translation,
                    "lidar2ego_rotation": lidar2ego_rotation,
                    "timestamp": timestamp,
                }

                # camera-obtain 6 image's information per frame
                camera_types = [
                    "cam_back_120",
                    "cam_back_left_120",
                    "cam_back_right_120",
                    # "cam_front_30",
                    "cam_front_70_left",
                    # "cam_front_70_right",
                    "cam_front_left_120",
                    "cam_front_right_120"
                ]
                for cam in camera_types:
                    # 获取相机内外参，旋转矩阵等...
                    cam_info = dict(
                        camera_path=cam_path,
                        lidar2img=lidar2img,
                        camera_intrinsics=cam_intrinsics,
                        # camera2lidar_rotation=cam2lidar_rotation,
                        # camera2lidar_translation=cam2lidar_translation,
                        lidar2cam_rotation=lidar2cam_rotation,
                        lidar2cam_translation=lidar2cam_translation
                    )
                    info["cams"].update({cam: cam_info})

                # obtain annotation
                if not test:
                    # 获取标签数据...
                    info["gt_boxes"] = gt_boxes
                    info["gt_names"] = names
                    # 获取其他数据...
                    info["num_lidar_pts"] = np.array([a["num_lidar_pts"] for a in annotations])
                    info["is_2d_visible"] = np.array([a["is_2d_visible"] for a in annotations])

                if jsondata['scene_id'].strip('.json') in train_scenes:
                    train_nusc_infos.append(info)
                if jsondata['scene_id'].strip('.json') in val_scenes:
                    val_nusc_infos.append(info)

    return train_nusc_infos, val_nusc_infos

```

### 1.3.保存点云真值（可选）

```python
# projects/Meg_dataset/tools/data_convert/create_gt_database.py
import pickle
from os import path as osp
import mmcv
import numpy as np
from mmcv import track_iter_progress
from mmdet3d.core.bbox import box_np_ops as box_np_ops
from mmdet3d.datasets import build_dataset


def create_groundtruth_database(
    dataset_class_name,
    data_path,
    info_prefix,
    info_path=None,

    used_classes=None,
    database_save_path=None,
    db_info_save_path=None,

):

    print(f"Create GT Database of {dataset_class_name}")
    dataset_cfg = dict(
        type=dataset_class_name, dataset_root=data_path, ann_file=info_path
    )

    if dataset_class_name == 'MegDataset':
        dataset_cfg.update(
            use_valid_flag=False,
            pipeline=[
                dict(
                    type="LoadPointsFromFile",
                    coord_type="LIDAR",
                    load_dim=5,
                    use_dim=5,
                ),
                dict(
                    type="LoadAnnotations3D",
                    with_bbox_3d=True,
                    with_label_3d=True
                ),
            ],
        )

    dataset = build_dataset(dataset_cfg)

    if database_save_path is None:
        database_save_path = osp.join(data_path, f"{info_prefix}_gt_database")
    if db_info_save_path is None:
        db_info_save_path = osp.join(data_path, f"{info_prefix}_dbinfos_train.pkl")
    mmcv.mkdir_or_exist(database_save_path)
    all_db_infos = dict()

    group_counter = 0
    for j in track_iter_progress(list(range(len(dataset)))):
        input_dict = dataset.get_data_info(j)
        dataset.pre_pipeline(input_dict)
        example = dataset.pipeline(input_dict)
        annos = example["ann_info"]
        image_idx = example["sample_idx"]
        points = example["points"].tensor.numpy()
        gt_boxes_3d = annos["gt_bboxes_3d"].tensor.numpy()
        names = annos["gt_names"]
        group_dict = dict()
        if "group_ids" in annos:
            group_ids = annos["group_ids"]
        else:
            group_ids = np.arange(gt_boxes_3d.shape[0], dtype=np.int64)
        difficulty = np.zeros(gt_boxes_3d.shape[0], dtype=np.int32)
        if "difficulty" in annos:
            difficulty = annos["difficulty"]

        num_obj = gt_boxes_3d.shape[0]
        point_indices = box_np_ops.points_in_rbbox(points, gt_boxes_3d)

        for i in range(num_obj):
            filename = f"{image_idx}_{names[i]}_{i}.bin"
            abs_filepath = osp.join(database_save_path, filename)
            rel_filepath = osp.join(f"{info_prefix}_gt_database", filename)

            # save point clouds and image patches for each object
            gt_points = points[point_indices[:, i]]
            gt_points[:, :3] -= gt_boxes_3d[i, :3]

            with open(abs_filepath, "w") as f:
                gt_points.tofile(f)

            if (used_classes is None) or names[i] in used_classes:
                db_info = {
                    "name": names[i],
                    "path": rel_filepath,
                    "image_idx": image_idx,
                    "gt_idx": i,
                    "box3d_lidar": gt_boxes_3d[i],
                    "num_points_in_gt": gt_points.shape[0],
                    "difficulty": difficulty[i],
                }
                local_group_id = group_ids[i]
                # if local_group_id >= 0:
                if local_group_id not in group_dict:
                    group_dict[local_group_id] = group_counter
                    group_counter += 1
                db_info["group_id"] = group_dict[local_group_id]
                if "score" in annos:
                    db_info["score"] = annos["score"][i]

                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]

    for k, v in all_db_infos.items():
        print(f"load {len(v)} {k} database infos")

    with open(db_info_save_path, "wb") as f:
        pickle.dump(all_db_infos, f)

```

> 上述代码目的是把数据集中的点云真值给扣出来，单独放在一个个bin文件中，可以作为点云数据的数据增强使用，在此项目中没有用到点云真值数据增强，可以不写。因此在上面的 `tools/create_data.py` 代码中，`create_groundtruth_database` 函数也做了删除注释。

### 1.4.创建自定义数据集类

```python
# projects/Meg_dataset/mmdet3d/datasets/meg_dataset.py
from typing import Any, Dict
import mmcv
import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.datasets.custom_3d import Custom3DDataset
import os.path as osp
import tempfile
from ...mmdet3d.evaluate.map import calculate_map


@DATASETS.register_module()
class MegDataset(Custom3DDataset):
    def __init__(
        self,
        ann_file,
        pipeline=None,
        dataset_root=None,
        object_classes=None,
        load_interval=1,
        with_velocity=False,
        modality=None,
        box_type_3d="LiDAR",
        filter_empty_gt=True,
        data_config=None,
        test_mode=False,
        use_valid_flag=False,
    ) -> None:
        # 设置数据集的基本信息...

    def get_cat_ids(self, idx):
        """Get category distribution of single scene.

        Args:
            idx (int): Index of the data_info.

        Returns:
            dict[list]: for each category, if the current scene
                contains such boxes, store a list containing idx,
                otherwise, store empty list.
        """
        info = self.data_infos[idx]
        if self.use_valid_flag:
            mask = info["valid_flag"]
            gt_names = set(info["gt_names"][mask])
        else:
            gt_names = set(info["gt_names"])

        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids

    def load_annotations(self, ann_file):
        # 从 ann_file 中读取一些基本信息，如 metas...
        return data_infos

    def get_data_info(self, index: int) -> Dict[str, Any]:
        info = self.data_infos[index]
        input_dict = dict(
            sample_idx=info['frame_id'],
            lidar_path=info["lidar_path"],
            sweeps=info["sweeps"],
            timestamp=info["timestamp"]
        )

        if self.modality["use_camera"]:
            for 6 个相机:
                # 从 6 个相机读取相机内外参、坐标转换矩阵...

            input_dict.update(
                dict(
                    image_paths=image_paths,
                    lidar2camera=lidar2cameras,
                    lidar2image=lidar2images,
                    lidar2image_1=lidar2image_1,
                    camera2ego=camera2ego,
                    camera2lidar=camera2lidars,
                    camera_intrinsics=cam_intrinsics
                )
            )

        # if not self.test_mode:
        # TODO (Haotian): test set submission.
        annos = self.get_ann_info(index)
        input_dict["ann_info"] = annos

        return input_dict

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        # 根据 index 获取对应的标签...
        info = self.data_infos[index]
        # filter out bbox containing no points...
        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
        )
        return anns_results

    def evaluate(
        self,
        results,
        metric="bbox",
        **kwargs
    ):
        metrics_dict = self.calc_metrics(results)  # 计算评价指标
        tmp_dir = tempfile.TemporaryDirectory()
        tmp_json = osp.join(tmp_dir.name, "metrics_summary.json")
        mmcv.dump(metrics_dict, tmp_json)

        tmp_dir.cleanup()
        return metrics_dict

    def calc_metrics(self, results, score_thr=0.5):
        # mAP 计算...
        metrics_summary = {
            'mAP': mAP,
        }

        return metrics_summary

```

> 上述代码中，可以暂时忽略 *evaluate 方法*和 *calc_metrics 方法*。这两个方法是在测试的时候计算 mAP 用，训练的时候没用。

### 1.5.处理数据成 pkl

首先编写 bash 脚本，修改 `ROOT_PATH_PROJ` 为自己的项目根目录，修改 `ROOT_PATH_DATASET` 为自定义数据集的目录。

```bash
# projects/Meg_dataset/bash_runner/create_data.sh
ROOT_PATH_PROJ='/path/to/your//bevfusion/'
ROOT_PATH_DATASET=${ROOT_PATH_PROJ}'data/meg_data/new_custom_data'
echo ${ROOT_PATH_DATASET}
python projects/Meg_dataset/tools/create_data.py meg --root-path ${ROOT_PATH_DATASET} --out-dir ${ROOT_PATH_DATASET} --extra-tag meg
```

然后可以开始处理数据。

```bash
sh projects/Meg_dataset/bash_runner/create_data.sh
```

> 完成后，会在数据集根目录下多出 2 个 pkl 文件。

## 2.重构模型文件

bevfusion 的原作者写代码不严谨，估计是为了方便，在 forward 中传入了一些毫无相关的参数。为了项目的简洁性，我们继承原作者的代码，但是删除不必要的参数代码。

### 2.1.重构 bevfusion 模型

```python
# projects/Meg_dataset/mmdet3d/models/bevfusion_simple.py
import torch
from mmcv.runner import auto_fp16, force_fp32
from torch.nn import functional as F
import logging
from mmdet3d.models import FUSIONMODELS
from mmdet3d.models.fusion_models.bevfusion import BEVFusion


@FUSIONMODELS.register_module()
class BEVFusionSimple(BEVFusion):
    """
    原版的 BevFusion 在 forward 里传入了一堆没用的参数。这里将其删除，只保留有用的参数。
    """
    @auto_fp16(apply_to=("img"))
    def forward(
            self,
            img,
            points,
            camera2ego,
            camera2lidar,
            camera_intrinsics,
            img_aug_matrix,
            lidar_aug_matrix,
            metas,
            gt_masks_bev=None,
            gt_bboxes_3d=None,
            gt_labels_3d=None,
            **kwargs,
    ):
        # 删除原版 bevfusion 没用的参数，并整理代码...

```

### 2.2.重构 lss_transform

```python
# projects/Meg_dataset/mmdet3d/models/lss_transform_simple.py
from mmcv.runner import force_fp32
from mmdet3d.models.builder import VTRANSFORMS
from mmdet3d.models.vtransforms.lss import LSSTransform

__all__ = ["LSSTransformSimple"]


@VTRANSFORMS.register_module()
class LSSTransformSimple(LSSTransform):
    """
    原版的 LSSTransform 继承 BaseTransform， BaseTransform 在 forward 里传入了一堆没用的参数。
    这里将其重写，只保留有用的参数。
    """

    @force_fp32()
    def forward(
        self,
        img,
        camera2lidar,
        camera_intrinsics,
        img_aug_matrix,
        lidar_aug_matrix
    ):
        # 删除原版 bevfusion 没用的参数，并整理代码...

```

### 2.3.重构 pillar_encoder 模块

```python
# projects/Meg_dataset/mmdet3d/models/pillar_encoder.py
from typing import Any, Dict
from torch import nn
from torch.nn import functional as F

from mmdet3d.models.builder import build_backbone
from mmdet.models import BACKBONES
from mmdet3d.models.backbones.pillar_encoder import PointPillarsEncoder

__all__ = ["PointPillarsEncoderWithConv4x", "ConvTest"]

@BACKBONES.register_module()
class PointPillarsEncoderWithConv4x(PointPillarsEncoder):
    """
    PointPillarsEncoder 的基础上增加 4 倍下采样，
    使得与 camera 模块输出的 feature map 匹配。
    """
    def __init__(
        self,
        pts_voxel_encoder: Dict[str, Any],
        pts_middle_encoder: Dict[str, Any],
        pts_conv_encoder: Dict[str, Any],
        **kwargs,
    ):
        super().__init__(pts_voxel_encoder, pts_middle_encoder, **kwargs,)
        self.pts_conv_encoder = build_backbone(pts_conv_encoder)

    def forward(self, feats, coords, batch_size, sizes):
        x = super().forward(feats, coords, batch_size, sizes)
        # 使用 resnet 进行下采样...

        return x

```

> 本文在 PointPillarEncoder 中进行了 4 倍下采样，原因是将点云的 feature map 大小变得和图像一样大。

### 2.4.重构 Centerhead

自定义数据集的 CenterHead 不需要速度信息，因此，注释这几行代码。

```python
# projects/Meg_dataset/mmdet3d/models/centerhead_without_vel.py
import torch
from mmcv.runner import force_fp32
from mmdet3d.core import draw_heatmap_gaussian, gaussian_radius
from mmdet3d.models.builder import HEADS
from mmdet3d.models.heads.bbox.centerpoint import CenterHead, clip_sigmoid


@HEADS.register_module()
class CenterHeadWithoutVel(CenterHead):
    """
    注释掉 CenterHead 的 vx,vy,vel 等速度信息
    """

```

> 上述 4 段代码，将模型进行了重写，并起了新的名称，在下面的 config 配置文件中，也需要修改成这些名称。

## 3.编写配置文件

bevfusion 原作者将配置文件做成了 yaml 格式，不伦不类。笔者还是喜欢 mmdetection3d 的那种 python 格式。

### 3.1.default_runtime 配置文件

```python
# projects/Meg_dataset/configs/_base_/default_runtime.py

checkpoint_config = dict(interval=1)

log_config = dict(
    interval=50,   # 多少批次 打印一次
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ]
)

seed = 0
deterministic = False
load_from = False
# load_from      = '/home/data/BEV_Detection/bevfusion-main_v1/pretrained/2022-11-03-test/epoch_16.pth'
resume_from = False
cudnn_benchmark = False
distributed = True
dist_params = dict(backend='nccl')

```

### 3.2.bevfusion 配置文件

```python
# projects/Meg_dataset/configs/bevfusion_c_l_meg.py
_base_ = ['./_base_/default_runtime.py']
custom_imports = dict(
    imports=[
        'projects.Meg_dataset.mmdet3d.datasets.meg_dataset',
        'projects.Meg_dataset.mmdet3d.models.lss_transform_simple',
        'projects.Meg_dataset.mmdet3d.models.bevfusion_simple',
        'projects.Meg_dataset.mmdet3d.models.pillar_encoder',
        'projects.Meg_dataset.mmdet3d.models.centerhead_without_vel',
    ],
    allow_failed_imports=False)


data_config = {
    'cams': ['cam_front_left_120', 'cam_front_70_left', 'cam_front_right_120',
             'cam_back_left_120', 'cam_back_120', 'cam_back_right_120']
}

root_path = '/home/daxiongpro/code/bevfusion/'
pretrained_path = root_path + 'pretrained/'
dataset_type = 'MegDataset'
dataset_root = root_path + 'data/meg_data/new_custom_data/'

gt_paste_stop_epoch = -1
reduce_beams = 32
load_dim = 5
use_dim = 5
load_augmented = False
max_epochs = 24

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.1, 0.1, 0.2]
image_size = [256, 704]

augment2d = {
    'resize': [[0.38, 0.55], [0.48, 0.48]],
    'rotate': [-5.4, 5.4],
    'gridmask': dict(prob=0.0, fixed_prob=True)
}
augment3d = {
    'scale': [0.95, 1.05],
    'rotate': [-0.3925, 0.3925],
    'translate': 0.0
}

object_classes = [
    'car', 'truck', 'bus', 'motorcycle', 'pedestrian', 'bicycle', 'cyclist',
    'tricycle'
]

model = dict(
    type='BEVFusionSimple',
    depth_gt=False,
    encoders=dict(
        lidar=dict(
            sparse_shape=[512, 512, 1],
            voxelize_reduce=False,
            voxelize=dict(
                max_num_points=20,
                point_cloud_range=point_cloud_range,
                voxel_size=[0.2, 0.2, 8],
                max_voxels=[30000, 60000],
            ),
            backbone=dict(
                type='PointPillarsEncoderWithConv4x',
                pts_conv_encoder=dict(
                    type='ConvTest',
                ),
                pts_voxel_encoder=dict(
                    type='PillarFeatureNet',
                    in_channels=5,
                    feat_channels=[64, 64],
                    with_distance=False,
                    point_cloud_range=point_cloud_range,
                    voxel_size=[0.2, 0.2, 8],
                    norm_cfg=dict(
                        type='BN1d',
                        eps=1.0e-3,
                        momentum=0.01,
                    ),
                ),
                pts_middle_encoder=dict(
                    type='PointPillarsScatter',
                    in_channels=64,
                    output_shape=[512, 512],
                    sparse_shape=[512, 512, 1],
                ),
            ),
        ),
        camera=dict(
            backbone=dict(
                pretrained=pretrained_path + 'resnet50-0676ba61.pth',
                type='ResNet',
                depth=50,
                num_stages=4,
                out_indices=[2, 3],
                frozen_stages=-1,
                norm_cfg=dict(type='BN', requires_grad=True),
                norm_eval=False,
                with_cp=True,
                style='pytorch'
            ),
            neck=dict(
                type='GeneralizedLSSFPN',
                in_channels=[1024, 2048],
                out_channels=512,
                start_level=0,
                num_outs=3,
                norm_cfg=dict(type='BN2d', requires_grad=True),
                act_cfg=dict(type='ReLU', inplace=True),
                upsample_cfg=dict(mode='bilinear', align_corners=False)
            ),
            vtransform=dict(
                type='LSSTransformSimple',
                in_channels=512,
                out_channels=80,
                image_size=image_size,
                feature_size=[image_size[0] // 16, image_size[1] // 16],
                xbound=[-51.2, 51.2, 0.8],
                ybound=[-51.2, 51.2, 0.8],
                zbound=[-10.0, 10.0, 20.0],
                dbound=[1.0, 60.0, 1.0],
                downsample=1
            )
        )
    ),
    fuser=dict(
        type='ConvFuser',
        in_channels=[80, 256],
        out_channels=256,
    ),
    heads=dict(
        object=dict(
            type='CenterHeadWithoutVel',
            in_channels=256,
            train_cfg=dict(
                point_cloud_range=point_cloud_range,
                grid_size=[1024, 1024, 1],
                voxel_size=voxel_size,
                out_size_factor=8,
                dense_reg=1,
                gaussian_overlap=0.1,
                max_objs=500,
                min_radius=2,
                code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            ),
            test_cfg=dict(
                post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                max_per_img=500,
                max_pool_nms=False,
                min_radius=[4, 12, 10, 1, 0.85, 0.175],
                score_threshold=0.1,
                out_size_factor=8,
                # voxel_size=voxel_size[:2],
                nms_type=['circle', 'rotate', 'rotate', 'circle', 'rotate', 'rotate'],
                pre_max_size=1000,
                post_max_size=83,
                nms_thr=0.2,
                nms_scale=[[1.0], [1.0, 1.0], [1.0, 1.0], [1.0], [1.0, 1.0], [2.5, 4.0]]
            ),
            tasks=[
                ["car"], ["truck"], ["bus"], ["motorcycle", "bicycle", "cyclist"],
                ["pedestrian"], ["tricycle"]
            ],
            common_heads=dict(
                reg=[2, 2], height=[1, 2], dim=[3, 2], rot=[2, 2]
            ),
            share_conv_channel=64,
            bbox_coder=dict(
                type='CenterPointBBoxCoder',
                pc_range=point_cloud_range,
                post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                max_num=500,
                score_threshold=0.1,
                out_size_factor=8,
                voxel_size=voxel_size[:2],
                code_size=7
            ),
            separate_head=dict(
                type='SeparateHead', init_bias=-2.19, final_kernel=3
            ),
            loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
            loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
            norm_bbox=True
        ),
        map=None
    ),
    decoder=dict(
        backbone=dict(
            type='GeneralizedResNet',
            in_channels=256,
            blocks=[[2, 128, 2], [2, 256, 2], [2, 512, 1]],
        ),
        neck=dict(
            type='LSSFPN', in_indices=[-1, 0], in_channels=[512, 128],
            out_channels=256, scale_factor=2
        )
    ),
)


train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=load_dim,
        use_dim=use_dim,
        reduce_beams=reduce_beams,
        load_augmented=load_augmented
    ),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False
    ),
    dict(
        type='ImageAug3D',
        final_dim=image_size,
        resize_lim=augment2d['resize'][0],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=augment2d['rotate'],
        rand_flip=True,
        is_train=True
    ),
    dict(
        type='GlobalRotScaleTrans',
        resize_lim=augment3d['scale'],
        rot_lim=augment3d['rotate'],
        trans_lim=augment3d['translate'],
        is_train=True
    ),
    dict(type='RandomFlip3D'),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=object_classes),
    dict(
        type='ImageNormalize',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    dict(
        type='GridMask', use_h=True, use_w=True, max_epoch=max_epochs,
        rotate=1, offset=False, ratio=0.5, mode=1,
        prob=augment2d['gridmask']['prob'],
        fixed_prob=augment2d['gridmask']['fixed_prob']
    ),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', classes=object_classes),
    dict(type='Collect3D', keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'],
         meta_keys=['camera_intrinsics', 'camera2ego', 'lidar2image', 'camera2lidar',
                    'lidar2camera', 'img_aug_matrix', 'lidar_aug_matrix']
         )
]


test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=load_dim,
        use_dim=use_dim,
        reduce_beams=reduce_beams,
        load_augmented=load_augmented
    ),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False
    ),
    dict(
        type='ImageAug3D',
        final_dim=image_size,
        resize_lim=augment2d['resize'][1],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[0.0, 0.0],
        rand_flip=False,
        is_train=False
    ),
    dict(
        type='GlobalRotScaleTrans',
        resize_lim=[1.0, 1.0],
        rot_lim=[0.0, 0.0],
        trans_lim=0.0,
        is_train=False
    ),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='ImageNormalize',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    dict(type='DefaultFormatBundle3D', classes=object_classes),
    dict(type='Collect3D', keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'],
         meta_keys=['camera_intrinsics', 'camera2ego', 'lidar2image', 'camera2lidar',
                    'lidar2camera', 'img_aug_matrix', 'lidar_aug_matrix']
         )
]


input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False
)

data = dict(
    samples_per_gpu=6,
    workers_per_gpu=8,
    train=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            dataset_root=dataset_root,
            ann_file=dataset_root + 'meg_infos_train.pkl',
            pipeline=train_pipeline,
            object_classes=object_classes,
            modality=input_modality,
            data_config=data_config,
            test_mode=False,
            use_valid_flag=False,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR')),
    val=dict(
        type=dataset_type,
        dataset_root=dataset_root,
        ann_file=dataset_root + "meg_infos_val.pkl",
        pipeline=test_pipeline,
        object_classes=object_classes,
        modality=input_modality,
        data_config=data_config,
        box_type_3d='LiDAR',
        test_mode=False
    ),
    test=dict(
        type=dataset_type,
        dataset_root=dataset_root,
        ann_file=dataset_root + "meg_infos_val.pkl",
        pipeline=test_pipeline,
        object_classes=object_classes,
        modality=input_modality,
        data_config=data_config,
        box_type_3d='LiDAR',
        test_mode=True
    )
)

# Optimizer
optimizer = dict(type='AdamW', lr=2e-4, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(policy='CosineAnnealing', min_lr_ratio=1.0e-5)
runner = dict(type='CustomEpochBasedRunner', max_epochs=max_epochs)
evaluation = dict(interval=24, pipeline=test_pipeline)

```

> 上述代码中，修改 root_path 为自己的仓库根目录。并同时修改数据集根目录等信息。

## 4.开始训练

由于 bevfusion 原作者使用 yaml 作为配置文件，笔者改回了 python 作为配置文件，因此，在 train.py 中，读取配置文件部分也要相应修改：

### 4.1.修改 train.py

```python
# projects/Meg_dataset/tools/train.py
# 其余代码与原版一致...


def main():
    dist.init()

    # 其余代码与原版一致...

    # configs.load(args.config, recursive=True)
    # configs.update(opts)

    # cfg = Config(recursive_eval(configs), filename=args.config)
    cfg = Config.fromfile(args.config)
    # 后面代码与原版一致...


```

### 4.2.编写训练脚本：

```bash
# projects/Meg_dataset/bash_runner/train.sh
DATE=$(date '+%Y-%m-%d_%H-%M-%S')
TRAIN_PY='projects/Meg_dataset/tools/train.py'
CONFIG_FILE='projects/Meg_dataset/configs/bevfusion_c_l_meg.py'
WORK_DIR="runs/${DATE}/"

torchpack dist-run -np 1 python ${TRAIN_PY} ${CONFIG_FILE} --run-dir ${WORK_DIR}
# torchpack dist-run -np 1 python -m debugpy --listen 8531 --wait-for-client ${TRAIN_PY} ${CONFIG_FILE}

```

> 修改上述代码的路径。

### 4.3.训练

在完成上述所有的代码后，即可进行训练：

```bash
sh projects/Meg_dataset/bash_runner/train.sh
```

## 5.可视化结果

对结果的可视化，可以使用 tools/visualize.py 脚本，但是也要作出一些修改。

### 5.1.修改 tools/visualize.py

```python
# projects/Meg_dataset/tools/visualize.py
# 其余代码与原版一致...

def main() -> None:
    # dist.init()
    # 其余代码与原版一致...

#     configs.load(args.config, recursive=True)
#     configs.update(opts)
#     cfg = Config(recursive_eval(configs), filename=args.config)

    cfg = Config.fromfile(args.config)

    # 其余代码与原版一致...

    # build the model and load checkpoint
    if args.mode == "pred":
        model = build_model(cfg.model)
        load_checkpoint(model, args.checkpoint, map_location="cpu")

        # model = MMDistributedDataParallel(
        #     model.cuda(),
        #     device_ids=[torch.cuda.current_device()],
        #     broadcast_buffers=False,
        # )
        model = MMDataParallel(model, device_ids=[0])
        model.eval()

    # 其余代码与原版一致...

```

### 5.2.创建 mmdet3d/core/visualize.py

```python
# projects/Meg_dataset/mmdet3d/core/visualize.py

OBJECT_PALETTE = {
    "car": (255, 158, 0),
    "truck": (255, 99, 71),
    "construction_vehicle": (233, 150, 70),
    "bus": (255, 69, 0),
    "trailer": (255, 140, 0),
    "barrier": (112, 128, 144),
    "motorcycle": (255, 61, 99),
    "bicycle": (220, 20, 60),
    "pedestrian": (0, 0, 230),
    "traffic_cone": (47, 79, 79),
    "tricycle": (220, 20, 60),  # 相比原版 mmdet3d 的 visualize 增加 tricycle
    "cyclist": (220, 20, 60)  # 相比原版 mmdet3d 的 visualize 增加 cyclist
}
    # 其余代码与原版一致



```

### 5.3.编写可视化脚本

```bash
# projects/Meg_dataset/bash_runner/vis.sh
VIS_PY='projects/Meg_dataset/tools/visualize.py'
CONFIG_FILE='projects/Meg_dataset/configs/bevfusion_c_l_meg.py'
CHECK_POINT='pretrained/2023-06-13_08-46-56/epoch_24.pth'
DEBUG_PY='-m debugpy --listen 8531 --wait-for-client'

python ${VIS_PY} ${CONFIG_FILE} --checkpoint ${CHECK_POINT}
# python ${DEBUG_PY} ${VIS_PY} ${CONFIG_FILE} --checkpoint ${CHECK_POINT}

```

### 5.4.数据可视化

```bash
sh projects/Meg_dataset/bash_runner/vis.sh
```

## 6.测试

训练完成后需要测试。传统的 nuscenes数据集有很多指标，NDS, mAP 等等。笔者使用 mAP 作为评价指标。

### 6.1.修改 test.py

与 4.1.小节一样，需要修改一些 config 读取相关的代码。

```python
# projects/Meg_dataset/tools/test.py
# 其余代码与原版一致...


def main():
    # 其余代码与原版一致...

    # configs.load(args.config, recursive=True)
    # cfg = Config(recursive_eval(configs), filename=args.config)
    cfg = Config.fromfile(args.config)
    print(cfg)
    # 其余代码与原版一致...

```

### 6.2.评价指标 mAP

在这个版本的 mmdetection3d 中，evaluate 代码都是写在数据集中的。可以参考 1.4.小节 *evaluate 方法*和 *calc_metrics 方法*。

在 1.4.小节的 meg_dataset.py 中，调用 map 模块，其实现如下：

```python
import numpy as np


def calculate_iou(box1, box2):
    # 计算两个3D框之间的IoU（Intersection over Union）
    # 这里假设box1和box2都是(x, y, z, w, h, d)形式的框，分别表示中心坐标和宽高深度
    # 返回IoU值

    # 计算两个框的边界坐标
    box1_min = (box1[0] - box1[3] / 2, box1[1] - box1[4] / 2, box1[2] - box1[5] / 2)
    box1_max = (box1[0] + box1[3] / 2, box1[1] + box1[4] / 2, box1[2] + box1[5] / 2)
    box2_min = (box2[0] - box2[3] / 2, box2[1] - box2[4] / 2, box2[2] - box2[5] / 2)
    box2_max = (box2[0] + box2[3] / 2, box2[1] + box2[4] / 2, box2[2] + box2[5] / 2)

    # 计算两个框的相交区域的体积
    intersect_min = np.maximum(box1_min, box2_min)
    intersect_max = np.minimum(box1_max, box2_max)
    intersect_size = np.maximum(0, intersect_max - intersect_min)

    # 计算两个框的并集区域的体积
    # box1_size = np.maximum(0, box1_max - box1_min)
    box1_size = (
        np.maximum(0, box1_max[0] - box1_min[0]),
        np.maximum(0, box1_max[1] - box1_min[1]),
        np.maximum(0, box1_max[2] - box1_min[2]))

    # box2_size = np.maximum(0, box2_max - box2_min)
    box2_size = (
        np.maximum(0, box2_max[0] - box2_min[0]),
        np.maximum(0, box2_max[1] - box2_min[1]),
        np.maximum(0, box2_max[2] - box2_min[2]))

    # union_size = box1_size + box2_size - intersect_size
    union_size = (
        box1_size[0] + box2_size[0] - intersect_size[0],
        box1_size[1] + box2_size[1] - intersect_size[1],
        box1_size[2] + box2_size[2] - intersect_size[2])

    # 计算IoU值
    iou = np.prod(intersect_size) / np.prod(union_size)
    return iou


def calculate_ap(recall, precision):
    # 计算单类别的平均精度（AP）
    # 输入为召回率（recall）和精确率（precision）数组
    # 返回AP值

    recall = np.array(recall).flatten()
    precision = np.array(precision).flatten()

    # 将召回率和精确率数组进行插值，以确保召回率是单调递增的
    recall_interp = np.linspace(0, 1, 101)
    precision_interp = np.interp(recall_interp, recall, precision)

    # 计算AP值
    ap = np.mean(precision_interp)
    return ap


def calculate_map(gt_boxes, pred_boxes, iou_threshold=0.5):
    """

    gt_boxes = [
        [(0, 0, 0, 0, 0, 0, 0), (1, 2, 3, 4, 5, 6), (2, 3, 4, 5, 6, 7), (1, 2, 3, 4, 5, 6)],  # 类别0的真实框
        [(0, 0, 0, 0, 0, 0, 0), (2, 3, 4, 5, 6, 7), (3, 4, 5, 6, 7, 8)],   # 类别1的真实框
        [(0, 0, 0, 0, 0, 0, 0)]
    ]

    pred_boxes = [
        [(0, 0, 0, 0, 0, 0, 0), (1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 0.9), (2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 0.8)],  # 类别0的预测框
        [(0, 0, 0, 0, 0, 0, 0), (2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 0.7), (3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 0.6)],   # 类别1的预测框
        [(0, 0, 0, 0, 0, 0, 0)]
    ]
    """
    # 计算3D目标检测的平均精度（mAP）
    # 输入为真实框（gt_boxes）和预测框（pred_boxes）列表，每个框都是(x, y, z, w, h, d)形式的元组
    # iou_threshold是IoU阈值，用于判断预测框和真实框之间的匹配关系
    # 返回mAP值

    num_classes = len(gt_boxes)  # 类别数
    aps = []  # 每个类别的AP值列表

    for class_idx in range(num_classes):
        gt_class_boxes = gt_boxes[class_idx]  # 当前类别的真实框列表
        pred_class_boxes = pred_boxes[class_idx]  # 当前类别的预测框列表

        num_gt_boxes = len(gt_class_boxes)
        num_pred_boxes = len(pred_class_boxes)

        # 初始化匹配矩阵
        match_matrix = np.zeros((num_pred_boxes, num_gt_boxes))

        # 对每个预测框和真实框计算IoU，并进行匹配
        for pred_idx, pred_box in enumerate(pred_class_boxes):
            for gt_idx, gt_box in enumerate(gt_class_boxes):
                iou = calculate_iou(pred_box, gt_box)
                if iou > iou_threshold:
                    match_matrix[pred_idx, gt_idx] = 1

        # 计算每个预测框的置信度得分
        scores = [box[6] for box in pred_class_boxes]

        # 根据置信度得分对匹配矩阵进行排序
        sorted_indices = np.argsort(scores)[::-1]
        match_matrix = match_matrix[sorted_indices, :]

        # 计算召回率和精确率
        true_positives = np.cumsum(match_matrix, axis=0)
        false_positives = np.cumsum(1 - match_matrix, axis=0)
        recall = true_positives / num_gt_boxes
        precision = true_positives / (true_positives + false_positives)

        # 计算AP值
        ap = calculate_ap(recall, precision)
        aps.append(ap)

    aps_filter_0 = list(filter(lambda x: x != 0, aps))  # 去掉 0
    # 计算mAP值
    if len(aps_filter_0) == 0:
        return 0.0
    else:
        mAP = np.mean(aps_filter_0)
        return mAP


def main():
    # 生成示例数据
    gt_boxes = [
        [(1, 2, 3, 4, 5, 6), (2, 3, 4, 5, 6, 7), (1, 2, 3, 4, 5, 6)],  # 类别0的真实框
        [(2, 3, 4, 5, 6, 7), (3, 4, 5, 6, 7, 8)],   # 类别1的真实框
        [(0, 0, 0, 0, 0, 0, 0)]
    ]

    pred_boxes = [
        [(1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 0.9), (2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 0.8)],  # 类别0的预测框
        [(2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 0.7), (3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 0.6)],   # 类别1的预测框
        [(0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0)]
    ]

    # 计算mAP
    mAP = calculate_map(gt_boxes, pred_boxes)

    # 输出结果
    print("mAP:", mAP)


if __name__ == '__main__':
    main()

```

### 6.3.编写测试脚本

```bash
TEST_PY='projects/Meg_dataset/tools/test.py'
CONFIG_FILE='projects/Meg_dataset/configs/bevfusion_c_l_meg.py'
PTH='pretrained/2023-06-13_08-46-56/epoch_24.pth'
EVAL='map'

# torchpack dist-run -np 1 python ${TEST_PY} ${CONFIG_FILE} ${PTH} --eval ${EVAL}
torchpack dist-run -np 1 python -m debugpy --listen 8531 --wait-for-client ${TEST_PY} ${CONFIG_FILE} ${PTH} --eval ${EVAL}
```

> 修改上述代码的路径。

### 6.4.测试集测试

```bash
sh projects/Meg_dataset/bash_runner/test.sh
```

## 日期

* 2024/01/05：删除部分代码并添加注释
* 2023/06/26：创作日期
