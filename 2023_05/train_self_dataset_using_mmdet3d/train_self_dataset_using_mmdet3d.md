# 使用 mmdet3d 框架训练自定义数据集

mmdet3d 框架是商汤的 3D 目标检测工具箱，可以训练、评测、可视化。详情：[mmdet3d 官网](https://github.com/open-mmlab/mmdetection3d) 、[官方文档](https://mmdetection3d.readthedocs.io/zh_CN/latest/tutorials/customize_dataset.html)。官方文档中，提供了 3 种不同的方式：支持新的数据格式、将新数据的格式转换为现有数据的格式、将新数据集的格式转换为一种当前可支持的中间格式。这三种方式大同小异。本质都是 4 步：

1. 将数据集预处理成 pkl 文件
2. 创建自定义数据集类，并在数据集中读取并解析 pkl 文件
3. 创建训练的配置文件
4. 创建评价类 Metric

其中，第 4 步在训练的过程中可有可无，在验证和测试的时候需要。

为了简便，本文只使用激光雷达数据作为例子，自定义数据集的名称叫 "meg"。使用的模型为 3DSSD 。

## 可视化数据集

可视化数据集的目的是看一下数据集的内容是否正确，GT bbox 是否能够准确框出场景中的障碍物信息。方法参考 [open3d 显示 pointcloud 和 bbox](2023_05/show_pointcloud/show_pointcloud_and_bbox.md) 。

## 数据集预处理成 pkl 文件

mmdet3d 官方提供的数据集都会先预处理成 pkl 文件，文件内保存一些数据集的基本信息，如数据存放路径，数据真值(bboxes、labels)， 相机内外参等。通常，自定义的数据集的 annotation info 保存格式可能有很多，如 json、yaml 等。我们需要将自定义的数据集的 anno 信息保存成 pkl 文件。这里可以模仿 nuscenes 数据集的创建过程，步骤如下：

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

上述代码中，主函数先读取参数，然后最关键的就是这一行：`meg_converter.create_meg_infos(root_path, info_prefix)`

### 编写处理 pkl 代码

create_meg_info 的定义以及实现如下：

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
    1. 读取自定义数据集的 anno info 文件
    2. 将自定义的 anno info 信息处理成 dict
    3. 将 dict 存入 train_infos 和 val_infos 列表
    """

    return train_infos, val_infos

```

上述代码中，class_names 更换为自定义数据集的类别。然后编写 `_fill_trainval_infos(root_path)` 函数，方法见函数内注释。运行 create_data.py 文件后，就会在数据集根目录下保存 pkl 文件。

### 查看处理好的 pkl 文件

在处理好 anno info 的 pkl 文件后，可以读取 pkl 文件查看。读取方法：

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


## 编写配置文件

## 日期

2023/05/25：创作日期
