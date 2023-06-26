# bevfusion 训练自定义数据集

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
    """
    划分训练集和测试集
    """
    p = osp.join(root_path, 'jsons')
    all_scenes = os.listdir(p)
    all_scenes = [scenes.split('.')[0] for scenes in all_scenes]

    test_num = math.floor(len(all_scenes) / 10)  # 取 1/10 场景为测试
    train_num = len(all_scenes) - test_num
    train_scenes = all_scenes[:train_num]
    val_scenes = all_scenes[train_num:]

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
        # dataset json
        with open(os.path.join(root_path, 'jsons', scenes_json + '.json'), 'r') as jsonread:
            jsondata = json.load(jsonread)

        for sample in jsondata['frames']:
            if sample['is_key_frame']:
                print(scenes_json, '---', sample['frame_id'], ' is key frame')
                frame_id = sample['frame_id'] + sid * 10000
                assert not root_path.endswith('/'), "root_path should not end with an '/' sign"
                lidar_path = osp.join(root_path, sample['sensor_data']['fuser_lidar']['file_path'])
                # lidar_path = sample['sensor_data']['front_lidar']['file_path']
                timestamp = eval(sample['sensor_data']['fuser_lidar']['timestamp']) * int(1000) * int(1000)
                lidar2ego_info = jsondata["calibrated_sensors"]['lidar_ego']['extrinsic']['transform']
                lidar2ego_translation = np.array([lidar2ego_info['translation']['x'],
                                                  lidar2ego_info['translation']['y'],
                                                  lidar2ego_info['translation']['z']])
                lidar2ego_rotation = np.array(Quaternion([lidar2ego_info['rotation']['w'],
                                                          lidar2ego_info['rotation']['x'],
                                                          lidar2ego_info['rotation']['y'],
                                                          lidar2ego_info['rotation']['z']]
                                                         ).rotation_matrix)
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
                    cam_path = osp.join(root_path, sample['sensor_data'][cam]['file_path'])
                    cam_intrinsics = np.array(
                        jsondata["calibrated_sensors"][cam]["intrinsic"]["K"],
                    )

                    lidar2cam_translation = np.array([
                        jsondata["calibrated_sensors"][cam]["extrinsic"]['transform']['translation']['x'],
                        jsondata["calibrated_sensors"][cam]["extrinsic"]['transform']['translation']['y'],
                        jsondata["calibrated_sensors"][cam]["extrinsic"]['transform']['translation']['z']]
                    )
                    lidar2cam_rotation = np.array(Quaternion([
                        jsondata["calibrated_sensors"][cam]["extrinsic"]['transform']['rotation']['w'],
                        jsondata["calibrated_sensors"][cam]["extrinsic"]['transform']['rotation']['x'],
                        jsondata["calibrated_sensors"][cam]["extrinsic"]['transform']['rotation']['y'],
                        jsondata["calibrated_sensors"][cam]["extrinsic"]['transform']['rotation']['z']]
                    ).rotation_matrix)
                    cam2lidar_rotation = np.linalg.inv(lidar2cam_rotation)
                    cam2lidar_translation = np.dot(cam2lidar_rotation, -lidar2cam_translation.T)

                    T_lidar_to_pixel = np.array(jsondata["calibrated_sensors"][cam]
                                                ['T_lidar_to_pixel'], dtype=np.float32)
                    lidar2img = np.eye(4).astype(np.float32)
                    lidar2img[:3, :3] = T_lidar_to_pixel[:3, :3]
                    lidar2img[:3, 3] = T_lidar_to_pixel[:3, 3:].T
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
                    annotations = sample['labels']
                    locs = np.array([[box['xyz_lidar']['x'],
                                      box['xyz_lidar']['y'],
                                      box['xyz_lidar']['z']] for box in annotations]
                                    ).reshape(-1, 3)
                    dims = np.array([[box['lwh']['w'],
                                      box['lwh']['l'],
                                      box['lwh']['h']] for box in annotations]
                                    ).reshape(-1, 3)
                    locs[:, 2] = locs[:, 2] - dims[:, 2] / 2.0  # mmdet3d 中以底边中心为中心点
                    rots = np.array([Quaternion([box['angle_lidar']['w'],
                                                 box['angle_lidar']['x'],
                                                 box['angle_lidar']['y'],
                                                 box['angle_lidar']['z']]
                                                ).yaw_pitch_roll[0] for box in annotations]
                                    ).reshape(-1, 1)

                    names = [box['category'] for box in annotations]
                    for i in range(len(names)):
                        if names[i] in MegDataset.NameMapping:
                            names[i] = MegDataset.NameMapping[names[i]]
                    names = np.array(names)

                    # we need to convert rot to SECOND format.
                    gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
                    assert len(gt_boxes) == len(annotations), f"{len(gt_boxes)}, {len(annotations)}"
                    info["gt_boxes"] = gt_boxes
                    info["gt_names"] = names
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
    NameMapping = {
        "小汽车": "car",
        "汽车": "car",
        "货车": "truck",
        "工程车": "construction_vehicle",
        "巴士": "bus",
        "摩托车": "motorcycle",
        "自行车": "bicycle",
        "三轮车": "tricycle",
        "骑车人": "cyclist",
        "骑行的人": "cyclist",
        "人": "pedestrian",
        "行人": "pedestrian",
        "其它": "other",
        "残影": "ghost",
        "蒙版": "masked_area",
        "其他": "other",
        "拖挂": "other",
        "锥桶": "traffic_cone",
        "防撞柱": "traffic_cone"
    }

    ErrNameMapping = {
        "trans_err": "mATE",
        "scale_err": "mASE",
        "orient_err": "mAOE",
        "vel_err": "mAVE",
        "attr_err": "mAAE",
    }

    CLASSES = (
        "car",
        "truck",
        "construction_vehicle",
        "bus",
        "motorcycle",
        "bicycle",
        "tricycle",
        "cyclist",
        "pedestrian",
        "other",
        "ghost",
        "masked_area",
        "traffic_cone"
    )

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
        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag
        super().__init__(
            dataset_root=dataset_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=object_classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
        )

        self.with_velocity = with_velocity
        self.data_config = data_config

        if self.modality is None:
            self.modality = dict(
                use_camera=False,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )

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
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file)
        data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
        data_infos = data_infos[:: self.load_interval]
        self.metadata = data["metadata"]
        self.version = self.metadata["version"]
        return data_infos

    def get_data_info(self, index: int) -> Dict[str, Any]:
        info = self.data_infos[index]
        input_dict = dict(
            sample_idx=info['frame_id'],
            lidar_path=info["lidar_path"],
            sweeps=info["sweeps"],
            timestamp=info["timestamp"]
        )
        # lidar to ego transform
        # lidar2ego          = np.eye(4).astype(np.float32)
        # lidar2ego[:3, :3]  = info["lidar2ego_rotation"]
        # lidar2ego[:3, 3]   = info["lidar2ego_translation"]
        # input_dict["lidar2ego"]  = lidar2ego

        if self.modality["use_camera"]:
            image_paths = []
            lidar2cameras = []
            lidar2images = []
            camera2ego = []
            cam_intrinsics = []
            camera2lidars = []
            lidar2image_1 = []

            # info["cams"]:['cam_back_120', 'cam_back_left_120', 'cam_back_right_120', 'cam_front_30', 'cam_front_70_left',
            # 'cam_front_70_right', 'cam_front_left_120', 'cam_front_right_120']
            for camera_type, camera_info in info["cams"].items():
                if camera_type in self.data_config['cams']:
                    image_paths.append(camera_info["camera_path"])

                    # lidar to camera transform
                    lidar2camera_rt = np.eye(4).astype(np.float32)
                    lidar2camera_rt[:3, :3] = camera_info['lidar2cam_rotation']
                    lidar2camera_rt[:3, 3] = camera_info['lidar2cam_translation']
                    lidar2cameras.append(lidar2camera_rt)

                    # camera to lidar transform
                    camera2lidar_rt = np.array(np.linalg.inv(lidar2camera_rt), dtype=np.float32)
                    camera2lidars.append(camera2lidar_rt)

                    # camera intrinsics
                    camera_intrinsics = np.eye(4).astype(np.float32)
                    camera_intrinsics[:3, :3] = camera_info['camera_intrinsics']
                    cam_intrinsics.append(camera_intrinsics)

                    # lidar to image transform
                    lidar2img = camera_info['lidar2img']
                    lidar2images.append(lidar2img)

                    # lidar to image transform
                    cam_intrins = np.eye(4).astype(np.float32)
                    cam_intrins_1 = camera_info['camera_intrinsics']
                    cam_intrins_1[:2, :2] = cam_intrins_1[:2, :2] / 2.0
                    cam_intrins_1[:2, 2:] = cam_intrins_1[:2, 2:] / 2.0
                    cam_intrins[:3, :3] = cam_intrins_1
                    lidar2image = cam_intrins @ lidar2camera_rt
                    lidar2image_1.append(lidar2image)

                    # camera to ego transformn
                    cam2ego = np.eye(4).astype(np.float32)
                    camera2ego.append(cam2ego)

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
        info = self.data_infos[index]
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info["valid_flag"]
        else:
            mask = info["num_lidar_pts"] > 0
        gt_bboxes_3d = info["gt_boxes"][mask]
        gt_names_3d = info["gt_names"][mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)

        gt_labels_3d = np.array(gt_labels_3d)
        label_mask = gt_labels_3d >= 0
        gt_labels_3d = gt_labels_3d[label_mask]
        gt_bboxes_3d = gt_bboxes_3d[label_mask]  # TODO 过滤非指定类别的信息

        if self.with_velocity:
            gt_velocity = info["gt_velocity"][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        # haotian: this is an important change: from 0.5, 0.5, 0.5 -> 0.5, 0.5, 0
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0)
        ).convert_to(self.box_mode_3d)

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

        # self.metric_table(tmp_json)  # 表格形式输出评价指标
        # self.metric_dict(tmp_json)  # 字典形式输出评价指标

        tmp_dir.cleanup()
        return metrics_dict

    def calc_metrics(self, results, score_thr=0.5):
        #  results[0]: dict_keys(['boxes_3d', 'scores_3d', 'labels_3d'])
        mAP_list = []  # 存放每一帧的 mAP
        for frame_i, (frame_gt, frame_pred) in enumerate(zip(self.data_infos, results)):
            gt_boxes_list = [[(0, 0, 0, 0, 0, 0, 0)] for i in range(len(self.CLASSES))]
            pred_boxes_list = [[(0, 0, 0, 0, 0, 0, 0)] for i in range(len(self.CLASSES))]
            for gt_box, gt_label in zip(frame_gt['gt_boxes'], frame_gt['gt_names']):
                if str(gt_label) != 'masked_area':  # 过滤掉对象车道蒙板
                    gt_label_idx = self.CLASSES.index(str(gt_label))
                    gt_boxes_list[gt_label_idx].append(gt_box)

            for pred_box, pred_score, pred_label_idx in zip(frame_pred['boxes_3d'], frame_pred['scores_3d'], frame_pred['labels_3d']):
                if pred_score >= score_thr:
                    pred_boxes_list[int(pred_label_idx)].append(pred_box)

            # 计算单帧 mAP
            mAP = calculate_map(gt_boxes_list, pred_boxes_list, iou_threshold=0.5)
            print("frame_{} mAP is {}:".format(frame_i, mAP))
            mAP_list.append(mAP)

        mAP_list_filter_0 = list(filter(lambda x: x != 0, mAP_list))  # 去掉 0
        mAP = np.mean(mAP_list_filter_0)
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
ROOT_PATH_PROJ='/home/daxiongpro/code/bevfusion/'
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

    def extract_camera_features(
            self,
            x,
            camera2lidar,
            camera_intrinsics,
            img_aug_matrix,
            lidar_aug_matrix
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)

        x = self.encoders["camera"]["backbone"](x)
        x = self.encoders["camera"]["neck"](x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        logging.info("ViewTransformer is LSSTransform_Img")
        x = self.encoders["camera"]["vtransform"](
            x,
            camera2lidar,
            camera_intrinsics,
            img_aug_matrix,
            lidar_aug_matrix
        )
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.encoders["lidar"]["voxelize"](res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats = feats.contiguous()

        return feats, coords, sizes

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
        features = []
        # for sensor in self.encoders:
        for sensor in (
                self.encoders if self.training else list(self.encoders.keys())[::-1]
        ):
            if sensor == "camera":
                feature = self.extract_camera_features(
                    img,
                    camera2lidar,
                    camera_intrinsics,
                    img_aug_matrix,
                    lidar_aug_matrix,
                )
            elif sensor == "lidar":
                feature = self.extract_lidar_features(points)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")
            features.append(feature)

        if not self.training:
            # avoid OOM
            features = features[::-1]

        if self.fuser is not None:
            x = self.fuser(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        batch_size = x.shape[0]

        x = self.decoder["backbone"](x)
        x = self.decoder["neck"](x)

        if self.training:
            outputs = {}
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                elif type == "map":
                    losses = head(x, gt_masks_bev)
                else:
                    raise ValueError(f"unsupported head: {type}")
                for name, val in losses.items():
                    if val.requires_grad:
                        outputs[f"loss/{type}/{name}"] = val * self.loss_scale[type]
                    else:
                        outputs[f"stats/{type}/{name}"] = val
            return outputs
        else:
            outputs = [{} for _ in range(batch_size)]
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    bboxes = head.get_bboxes(pred_dict, metas)
                    for k, (boxes, scores, labels) in enumerate(bboxes):
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                            }
                        )
                elif type == "map":
                    logits = head(x)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "masks_bev": logits[k].cpu(),
                                "gt_masks_bev": gt_masks_bev[k].cpu(),
                            }
                        )
                else:
                    raise ValueError(f"unsupported head: {type}")
            return outputs

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
        intrins = camera_intrinsics[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]

        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans
        )

        x = self.get_cam_feats(img)
        x = self.bev_pool(geom, x)
        return x

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


class RestNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        output = self.conv1(x)
        output = F.relu(self.bn1(output))
        output = self.conv2(output)
        output = self.bn2(output)
        return F.relu(x + output)


class RestNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetDownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.extra = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        extra_x = self.extra(x)
        output = self.conv1(x)
        out = F.relu(self.bn1(output))
        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(extra_x + out)


@BACKBONES.register_module()
class ConvTest(nn.Module):
    def __init__(self):
        super(ConvTest, self).__init__()

        self.layer1 = nn.Sequential(RestNetBasicBlock(64, 64, 1),
                                    RestNetBasicBlock(64, 64, 1))

        self.layer2 = nn.Sequential(RestNetDownBlock(64, 128, [2, 1]),
                                    RestNetBasicBlock(128, 128, 1))

        self.layer3 = nn.Sequential(RestNetDownBlock(128, 256, [2, 1]),
                                    RestNetBasicBlock(256, 256, 1))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out


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
        x = self.pts_conv_encoder(x)

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

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d):
        """Generate training targets for a single sample.
        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.
        Returns:
            tuple[list[torch.Tensor]]: Tuple of target including \
                the following results in order.
                - list[torch.Tensor]: Heatmap scores.
                - list[torch.Tensor]: Ground truth boxes.
                - list[torch.Tensor]: Indexes indicating the position \
                    of the valid boxes.
                - list[torch.Tensor]: Masks indicating which boxes \
                    are valid.
        """
        device = gt_labels_3d.device
        gt_bboxes_3d = torch.cat(
            (gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]), dim=1
        ).to(device)
        max_objs = self.train_cfg["max_objs"] * self.train_cfg["dense_reg"]
        grid_size = torch.tensor(self.train_cfg["grid_size"])
        pc_range = torch.tensor(self.train_cfg["point_cloud_range"])
        voxel_size = torch.tensor(self.train_cfg["voxel_size"])

        feature_map_size = torch.div(
            grid_size[:2],
            self.train_cfg["out_size_factor"],
            rounding_mode="trunc",
        )

        # reorganize the gt_dict by tasks
        task_masks = []
        flag = 0
        for class_name in self.class_names:
            task_masks.append(
                [
                    torch.where(gt_labels_3d == class_name.index(i) + flag)
                    for i in class_name
                ]
            )
            flag += len(class_name)

        task_boxes = []
        task_classes = []
        flag2 = 0
        for idx, mask in enumerate(task_masks):
            task_box = []
            task_class = []
            for m in mask:
                task_box.append(gt_bboxes_3d[m])
                # 0 is background for each task, so we need to add 1 here.
                task_class.append(gt_labels_3d[m] + 1 - flag2)
            task_boxes.append(torch.cat(task_box, axis=0).to(device))
            task_classes.append(torch.cat(task_class).long().to(device))
            flag2 += len(mask)
        draw_gaussian = draw_heatmap_gaussian
        heatmaps, anno_boxes, inds, masks = [], [], [], []

        for idx, task_head in enumerate(self.task_heads):
            heatmap = gt_bboxes_3d.new_zeros(
                (len(self.class_names[idx]), feature_map_size[1], feature_map_size[0])
            )

            anno_box = gt_bboxes_3d.new_zeros((max_objs, 8), dtype=torch.float32)

            ind = gt_labels_3d.new_zeros((max_objs), dtype=torch.int64)
            mask = gt_bboxes_3d.new_zeros((max_objs), dtype=torch.uint8)

            num_objs = min(task_boxes[idx].shape[0], max_objs)

            for k in range(num_objs):
                cls_id = task_classes[idx][k] - 1

                width = task_boxes[idx][k][3]
                length = task_boxes[idx][k][4]
                width = width / voxel_size[0] / self.train_cfg["out_size_factor"]
                length = length / voxel_size[1] / self.train_cfg["out_size_factor"]

                if width > 0 and length > 0:
                    radius = gaussian_radius(
                        (length, width), min_overlap=self.train_cfg["gaussian_overlap"]
                    )
                    radius = max(self.train_cfg["min_radius"], int(radius))

                    # be really careful for the coordinate system of
                    # your box annotation.
                    x, y, z = (
                        task_boxes[idx][k][0],
                        task_boxes[idx][k][1],
                        task_boxes[idx][k][2],
                    )

                    coor_x = (
                        (x - pc_range[0])
                        / voxel_size[0]
                        / self.train_cfg["out_size_factor"]
                    )
                    coor_y = (
                        (y - pc_range[1])
                        / voxel_size[1]
                        / self.train_cfg["out_size_factor"]
                    )

                    center = torch.tensor(
                        [coor_x, coor_y], dtype=torch.float32, device=device
                    )
                    center_int = center.to(torch.int32)

                    # throw out not in range objects to avoid out of array
                    # area when creating the heatmap
                    if not (
                        0 <= center_int[0] < feature_map_size[0]
                        and 0 <= center_int[1] < feature_map_size[1]
                    ):
                        continue

                    draw_gaussian(heatmap[cls_id], center_int[[1, 0]], radius)
                    new_idx = k
                    x, y = center_int[0], center_int[1]

                    assert (
                        x * feature_map_size[1] + y
                        < feature_map_size[0] * feature_map_size[1]
                    )

                    ind[new_idx] = x * feature_map_size[1] + y

                    mask[new_idx] = 1
                    # TODO: support other outdoor dataset
                    # vx, vy = task_boxes[idx][k][7:]
                    rot = task_boxes[idx][k][6]
                    box_dim = task_boxes[idx][k][3:6]
                    if self.norm_bbox:
                        box_dim = box_dim.log()
                    anno_box[new_idx] = torch.cat(
                        [
                            center - torch.tensor([x, y], device=device),
                            z.unsqueeze(0),
                            box_dim,
                            torch.sin(rot).unsqueeze(0),
                            torch.cos(rot).unsqueeze(0),
                            # vx.unsqueeze(0),
                            # vy.unsqueeze(0),
                        ]
                    )

            heatmaps.append(heatmap)
            anno_boxes.append(anno_box)
            masks.append(mask)
            inds.append(ind)
        return heatmaps, anno_boxes, inds, masks

    @force_fp32(apply_to=("preds_dicts"))
    def loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, **kwargs):
        """Loss function for CenterHead.
        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function.
        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        heatmaps, anno_boxes, inds, masks = self.get_targets(gt_bboxes_3d, gt_labels_3d)
        loss_dict = dict()
        for task_id, preds_dict in enumerate(preds_dicts):
            # heatmap focal loss
            preds_dict[0]["heatmap"] = clip_sigmoid(preds_dict[0]["heatmap"])
            num_pos = heatmaps[task_id].eq(1).float().sum().item()
            loss_heatmap = self.loss_cls(
                preds_dict[0]["heatmap"], heatmaps[task_id], avg_factor=max(num_pos, 1)
            )
            target_box = anno_boxes[task_id]
            # reconstruct the anno_box from multiple reg heads
            preds_dict[0]["anno_box"] = torch.cat(
                (
                    preds_dict[0]["reg"],
                    preds_dict[0]["height"],
                    preds_dict[0]["dim"],
                    preds_dict[0]["rot"],
                    # preds_dict[0]["vel"],
                ),
                dim=1,
            )

            # Regression loss for dimension, offset, height, rotation
            ind = inds[task_id]
            num = masks[task_id].float().sum()
            pred = preds_dict[0]["anno_box"].permute(0, 2, 3, 1).contiguous()
            pred = pred.view(pred.size(0), -1, pred.size(3))
            pred = self._gather_feat(pred, ind)
            mask = masks[task_id].unsqueeze(2).expand_as(target_box).float()
            isnotnan = (~torch.isnan(target_box)).float()
            mask *= isnotnan

            code_weights = self.train_cfg.get("code_weights", None)
            bbox_weights = mask * mask.new_tensor(code_weights)
            loss_bbox = self.loss_bbox(
                pred, target_box, bbox_weights, avg_factor=(num + 1e-4)
            )
            loss_dict[f"heatmap/task{task_id}"] = loss_heatmap
            loss_dict[f"bbox/task{task_id}"] = loss_bbox
        return loss_dict

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
import argparse
import copy
import os
import random
import time

import numpy as np
import torch
from mmcv import Config
from torchpack import distributed as dist
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs

from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger, convert_sync_batchnorm, recursive_eval


def main():
    dist.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    parser.add_argument("--run-dir", metavar="DIR", help="run directory")
    args, opts = parser.parse_known_args()

    # configs.load(args.config, recursive=True)
    # configs.update(opts)

    # cfg = Config(recursive_eval(configs), filename=args.config)
    cfg = Config.fromfile(args.config)

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(dist.local_rank())

    if args.run_dir is None:
        args.run_dir = auto_set_run_dir()
    else:
        set_run_dir(args.run_dir)
    cfg.run_dir = args.run_dir

    # dump config
    cfg.dump(os.path.join(cfg.run_dir, "configs.yaml"))

    # init the logger before other steps
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(cfg.run_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file)

    # log some basic info
    logger.info(f"Config:\n{cfg.pretty_text}")

    # set random seeds
    if cfg.seed is not None:
        logger.info(
            f"Set random seed to {cfg.seed}, "
            f"deterministic mode: {cfg.deterministic}"
        )
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if cfg.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    datasets = [build_dataset(cfg.data.train)]

    model = build_model(cfg.model)
    model.init_weights()
    if cfg.get("sync_bn", None):
        if not isinstance(cfg["sync_bn"], dict):
            cfg["sync_bn"] = dict(exclude=[])
        model = convert_sync_batchnorm(model, exclude=cfg["sync_bn"]["exclude"])

    logger.info(f"Model:\n{model}")
    train_model(
        model,
        datasets,
        cfg,
        distributed=True,
        validate=True,
        timestamp=timestamp,
    )


if __name__ == "__main__":
    main()

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
import argparse
import copy
import os

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel, MMDataParallel
from mmcv.runner import load_checkpoint
from torchpack import distributed as dist
from torchpack.utils.config import configs
# from torchpack.utils.tqdm import tqdm
from tqdm import tqdm

from mmdet3d.core import LiDARInstance3DBoxes
from projects.Meg_dataset.mmdet3d.core.visualize import visualize_camera, visualize_lidar, visualize_map
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model


def recursive_eval(obj, globals=None):
    if globals is None:
        globals = copy.deepcopy(obj)

    if isinstance(obj, dict):
        for key in obj:
            obj[key] = recursive_eval(obj[key], globals)
    elif isinstance(obj, list):
        for k, val in enumerate(obj):
            obj[k] = recursive_eval(val, globals)
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        obj = eval(obj[2:-1], globals)
        obj = recursive_eval(obj, globals)

    return obj


def main() -> None:
    # dist.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE")
    parser.add_argument("--mode", type=str, default="pred", choices=["gt", "pred"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--bbox-classes", nargs="+", type=int, default=None)
    parser.add_argument("--bbox-score", type=float, default=0.2)
    parser.add_argument("--map-score", type=float, default=0.5)
    parser.add_argument("--out-dir", type=str, default="viz")
    args, opts = parser.parse_known_args()

#     configs.load(args.config, recursive=True)
#     configs.update(opts)
#     cfg = Config(recursive_eval(configs), filename=args.config)

    cfg = Config.fromfile(args.config)

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(dist.local_rank())

    # build the dataloader
    dataset = build_dataset(cfg.data[args.split])
    dataflow = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=True,
        shuffle=False,
    )

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

    for data in tqdm(dataflow):
        metas = data["metas"].data[0][0]
        # name = "{}-{}".format(metas["timestamp"], metas["token"])
        name = "{}".format(metas["timestamp"])

        if args.mode == "pred":
            with torch.inference_mode():
                outputs = model(**data)

        if args.mode == "gt" and "gt_bboxes_3d" in data:
            bboxes = data["gt_bboxes_3d"].data[0][0].tensor.numpy()
            labels = data["gt_labels_3d"].data[0][0].numpy()

            if args.bbox_classes is not None:
                indices = np.isin(labels, args.bbox_classes)
                bboxes = bboxes[indices]
                labels = labels[indices]

            bboxes[..., 2] -= bboxes[..., 5] / 2
            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
        elif args.mode == "pred" and "boxes_3d" in outputs[0]:
            bboxes = outputs[0]["boxes_3d"].tensor.numpy()
            scores = outputs[0]["scores_3d"].numpy()
            labels = outputs[0]["labels_3d"].numpy()

            if args.bbox_classes is not None:
                indices = np.isin(labels, args.bbox_classes)
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]

            if args.bbox_score is not None:
                indices = scores >= args.bbox_score
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]

            # bboxes[..., 2] -= bboxes[..., 5] / 2
            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=7)
        else:
            bboxes = None
            labels = None

        if args.mode == "gt" and "gt_masks_bev" in data:
            masks = data["gt_masks_bev"].data[0].numpy()
            masks = masks.astype(np.bool)
        elif args.mode == "pred" and "masks_bev" in outputs[0]:
            masks = outputs[0]["masks_bev"].numpy()
            masks = masks >= args.map_score
        else:
            masks = None

        if "img" in data:
            for k, image_path in enumerate(metas["filename"]):
                image = mmcv.imread(image_path)
                visualize_camera(
                    os.path.join(args.out_dir, f"camera-{k}", f"{name}.png"),
                    image,
                    bboxes=bboxes,
                    labels=labels,
                    transform=metas["lidar2image"][k],
                    classes=cfg.object_classes,
                )

        if "points" in data:
            lidar = data["points"].data[0][0].numpy()
            visualize_lidar(
                os.path.join(args.out_dir, "lidar", f"{name}.png"),
                lidar,
                bboxes=bboxes,
                labels=labels,
                xlim=[cfg.point_cloud_range[d] for d in [0, 3]],
                ylim=[cfg.point_cloud_range[d] for d in [1, 4]],
                classes=cfg.object_classes,
            )

        if masks is not None:
            visualize_map(
                os.path.join(args.out_dir, "map", f"{name}.png"),
                masks,
                classes=cfg.map_classes,
            )


if __name__ == "__main__":
    main()

```

### 5.2.创建 mmdet3d/core/visualize.py

```python
# projects/Meg_dataset/mmdet3d/core/visualize.py
import copy
import os
from typing import List, Optional, Tuple

import cv2
import mmcv
import numpy as np
from matplotlib import pyplot as plt

# from ..bbox import LiDARInstance3DBoxes
from mmdet3d.core.bbox import LiDARInstance3DBoxes

__all__ = ["visualize_camera", "visualize_lidar", "visualize_map"]


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

MAP_PALETTE = {
    "drivable_area": (166, 206, 227),
    "road_segment": (31, 120, 180),
    "road_block": (178, 223, 138),
    "lane": (51, 160, 44),
    "ped_crossing": (251, 154, 153),
    "walkway": (227, 26, 28),
    "stop_line": (253, 191, 111),
    "carpark_area": (255, 127, 0),
    "road_divider": (202, 178, 214),
    "lane_divider": (106, 61, 154),
    "divider": (106, 61, 154),
}


def visualize_camera(
    fpath: str,
    image: np.ndarray,
    *,
    bboxes: Optional[LiDARInstance3DBoxes] = None,
    labels: Optional[np.ndarray] = None,
    transform: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
    color: Optional[Tuple[int, int, int]] = None,
    thickness: float = 4,
) -> None:
    canvas = image.copy()
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    if bboxes is not None and len(bboxes) > 0:
        corners = bboxes.corners
        num_bboxes = corners.shape[0]

        coords = np.concatenate(
            [corners.reshape(-1, 3), np.ones((num_bboxes * 8, 1))], axis=-1
        )
        transform = copy.deepcopy(transform).reshape(4, 4)
        coords = coords @ transform.T
        coords = coords.reshape(-1, 8, 4)

        indices = np.all(coords[..., 2] > 0, axis=1)
        coords = coords[indices]
        labels = labels[indices]

        indices = np.argsort(-np.min(coords[..., 2], axis=1))
        coords = coords[indices]
        labels = labels[indices]

        coords = coords.reshape(-1, 4)
        coords[:, 2] = np.clip(coords[:, 2], a_min=1e-5, a_max=1e5)
        coords[:, 0] /= coords[:, 2]
        coords[:, 1] /= coords[:, 2]

        coords = coords[..., :2].reshape(-1, 8, 2)
        for index in range(coords.shape[0]):
            name = classes[labels[index]]
            for start, end in [
                (0, 1),
                (0, 3),
                (0, 4),
                (1, 2),
                (1, 5),
                (3, 2),
                (3, 7),
                (4, 5),
                (4, 7),
                (2, 6),
                (5, 6),
                (6, 7),
            ]:
                cv2.line(
                    canvas,
                    coords[index, start].astype(np.int),
                    coords[index, end].astype(np.int),
                    color or OBJECT_PALETTE[name],
                    thickness,
                    cv2.LINE_AA,
                )
        canvas = canvas.astype(np.uint8)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    mmcv.imwrite(canvas, fpath)


def visualize_lidar(
    fpath: str,
    lidar: Optional[np.ndarray] = None,
    *,
    bboxes: Optional[LiDARInstance3DBoxes] = None,
    labels: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
    xlim: Tuple[float, float] = (-50, 50),
    ylim: Tuple[float, float] = (-50, 50),
    color: Optional[Tuple[int, int, int]] = None,
    radius: float = 15,
    thickness: float = 25,
) -> None:
    fig = plt.figure(figsize=(xlim[1] - xlim[0], ylim[1] - ylim[0]))

    ax = plt.gca()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect(1)
    ax.set_axis_off()

    if lidar is not None:
        plt.scatter(
            lidar[:, 0],
            lidar[:, 1],
            s=radius,
            c="white",
        )

    if bboxes is not None and len(bboxes) > 0:
        coords = bboxes.corners[:, [0, 3, 7, 4, 0], :2]
        for index in range(coords.shape[0]):
            name = classes[labels[index]]
            plt.plot(
                coords[index, :, 0],
                coords[index, :, 1],
                linewidth=thickness,
                color=np.array(color or OBJECT_PALETTE[name]) / 255,
            )

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    fig.savefig(
        fpath,
        dpi=10,
        facecolor="black",
        format="png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()


def visualize_map(
    fpath: str,
    masks: np.ndarray,
    *,
    classes: List[str],
    background: Tuple[int, int, int] = (240, 240, 240),
) -> None:
    assert masks.dtype == np.bool, masks.dtype

    canvas = np.zeros((*masks.shape[-2:], 3), dtype=np.uint8)
    canvas[:] = background

    for k, name in enumerate(classes):
        if name in MAP_PALETTE:
            canvas[masks[k], :] = MAP_PALETTE[name]
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    mmcv.imwrite(canvas, fpath)

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
import argparse
import copy
import os
import warnings

import mmcv
import torch
from torchpack.utils.config import configs
from torchpack import distributed as dist
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.utils import recursive_eval


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet test (and eval) a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--out", help="output result file in pickle format")
    parser.add_argument(
        "--fuse-conv-bn",
        action="store_true",
        help="Whether to fuse conv and bn, this will slightly increase"
        "the inference speed",
    )
    parser.add_argument(
        "--format-only",
        action="store_true",
        help="Format the output results without perform evaluation. It is"
        "useful when you want to format the result to a specific format and "
        "submit it to the test server",
    )
    parser.add_argument(
        "--eval",
        type=str,
        nargs="+",
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC',
    )
    parser.add_argument("--show", action="store_true", help="show results")
    parser.add_argument("--show-dir", help="directory where results will be saved")
    parser.add_argument(
        "--gpu-collect",
        action="store_true",
        help="whether to use gpu to collect results.",
    )
    parser.add_argument(
        "--tmpdir",
        help="tmp directory used for collecting results from multiple "
        "workers, available when gpu-collect is not specified",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="whether to set deterministic options for CUDNN backend.",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be kwargs for dataset.evaluate() function (deprecate), "
        "change to --eval-options instead.",
    )
    parser.add_argument(
        "--eval-options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be kwargs for dataset.evaluate() function",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            "--options and --eval-options cannot be both specified, "
            "--options is deprecated in favor of --eval-options"
        )
    if args.options:
        warnings.warn("--options is deprecated in favor of --eval-options")
        args.eval_options = args.options
    return args


def main():
    args = parse_args()
    dist.init()

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(dist.local_rank())

    assert args.out or args.eval or args.format_only or args.show or args.show_dir, (
        "Please specify at least one operation (save/eval/format/show the "
        'results / save the results) with the argument "--out", "--eval"'
        ', "--format-only", "--show" or "--show-dir"'
    )

    if args.eval and args.format_only:
        raise ValueError("--eval and --format_only cannot be both specified")

    if args.out is not None and not args.out.endswith((".pkl", ".pickle")):
        raise ValueError("The output file must be a pkl file.")

    # configs.load(args.config, recursive=True)
    # cfg = Config(recursive_eval(configs), filename=args.config)
    cfg = Config.fromfile(args.config)
    print(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop("samples_per_gpu", 1) for ds_cfg in cfg.data.test]
        )
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    distributed = True

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
    )

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    else:
        model.CLASSES = dataset.CLASSES

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
        outputs = multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f"\nwriting results to {args.out}")
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get("evaluation", {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                "interval",
                "tmpdir",
                "start",
                "gpu_collect",
                "save_best",
                "rule",
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == "__main__":
    main()

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

2023/06/26：创作日期
