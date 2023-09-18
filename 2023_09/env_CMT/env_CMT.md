# 环境调试——CMT：Cross Modal Transformer

CMT 的官方源代码已经在 [github](https://github.com/junjie18/CMT) 上发布。其源码也是基于 mmdet3d 框架。根据官方 README，笔者也测试了其在 nuscenes 上的精度，与论文所述一致。

## 官方源代码跑通(test 部分)

新手小白可能不太会 mmdet3d。笔者来个保姆级教程。建议完整看完本文，再动手实践，而不是边看边实践。

## 安装环境

* 新建 conda 环境

```bash
# conda env
conda create -n cmt python=3.8 -y
conda activate cmt
```

* pip 包安装

```bash
# pytorch 1.9.0+cu111
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
# mmlab
pip install mmcv-full==1.6.0
pip install mmdet==2.24.0
pip install mmsegmentation==0.29.1
pip install mmdet3d==1.0.0rc5
# spconv
pip install spconv-cu111==2.1.21
# flash-attn
pip install flash-attn==0.2.2
# cv2
pip install opencv-python
# nuscenes
pip install nuscenes-devkit
# numpy
pip install numpy==1.22.4
```

> 将以上代码逐行敲入命令行，环境安装完成。mmcv-full 安装时间很长。
>
> 注意：在安装环境之前删除所有 pip 缓存包 `rm -rf ~/.cache/pip` 。

## 数据集准备

在 CMT 根目录下：

```bash
mkdir data && cd data
ln -s /path/to/your/nuscenes/dataset/root/ ./nuscenes
cd ..
```

> 上述代码将 nuscenes 数据集软连接到项目中。修改 `/path/to/your/nuscenes/dataset/root/` 为你自己的 nuscenes 数据集在磁盘中的路径。

## 运行代码

官方作者使用的是分布式运行，即 `dist_train.py` 和 `dist_test.py`。分布式笔者暂时还不是很熟悉，不太会用。因此，笔者自己写了 `create_data.py` 和 `test.py` 的运行脚本。如果需要训练，也与之类似。

## 数据预处理

修改原作者的 create_data.sh，编写数据预处理脚本，然后运行之。

```bash
# create_data.sh
CREATE_DATA='tools/create_data.py'
DATASET_NAME='nuscenes'
ROOT_PATH_PROJ='/path/to/your/cmt/root'
ROOT_PATH="--root-path ${ROOT_PATH_PROJ}/data/nuscenes"
OUT_DIR="--out-dir ${ROOT_PATH_PROJ}/data/nuscenes"
EXTRA_TAG='--extra-tag nuscenes'
VERSION='--version v1.0'

python ${CREATE_DATA} ${DATASET_NAME} ${ROOT_PATH} ${OUT_DIR} ${EXTRA_TAG} ${VERSION}
```

> nuscenes 全集数据量巨大，数据预处理需要好几个小时。建议下班前跑起来，第二天早上处理完成。需要注意的是，数据预处理前，硬盘要腾出 100 个 G，不然磁盘容量不够，第二天一看白做。

## 下载 ckpt

* 新建文件夹：ckpt
* 根据[官方 README](https://github.com/junjie18/CMT)，下载 pth 权重，放到 ckpt 文件夹

## 测试数据

编写测试脚本，然后运行之。

```bash
# test.sh
TEST_PY='tools/test.py'
CONFIG_FILE='projects/configs/fusion/cmt_voxel0075_vov_1600x640_cbgs.py'
PTH='ckpt/voxel0075_vov_1600x640_epoch20.pth'
python ${TEST_PY} ${CONFIG_FILE} ${PTH} --eval bbox
```

> 不出意外可以看到测试效果，精度如原作者所述。

## 解决遇到的 bug

* SystemError: initialization of _internal failed without raising an exception

解决：重装 numba 库：

```bash
pip install -U numba  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

* ModuleNotFoundError: No module named 'projects'

解决：找不到 projects 包的原因是因为没有配置 python path。在 from projects.mmdet3d_plugin.datasets import CustomNuScenesDataset 之前插入代码：

```python
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
print(sys.path)
```

## 日期

2023/09/18：文章撰写日期
