# UniAD 加入 Lidar 模态

UniAD 加入 Lidar 模态相当于是在 BEVFormer 中加入 Lidar 模态。问题是如何加入？

## 读取历史帧数据

如何读取多帧图片数据？

使用的是 nuscenes_e2e_dataset - > prepare_train_data 方法，里面有个 for i in prev_indexs_list 循环。

## 处理点云数据

读取图片的时候，当前帧和历史帧都是 (6, c, h, w)，读 4 帧历史帧，可以和当前帧拼接成 (5, 6, c, h, w)。最终会形成 (bs, 5, 6, c, h, w)

点云数据不同，每帧点云点的个数不相同，(32692, 5), (32867, 5), ... 那么无法将多个历史帧拼接起来。

### 方法1

统一每一帧为 32768 个点，若当前帧点数不够，则补点: (0, 0, 0, 0, 0)。在 train_pipeline 中执行。若当前帧点数太多，则随机删除一些点。

### 方法2

使用 mmdet3d 框架的 Pointsample 类，在 train_pipeline 中加入 `dict(type='PointSample', num_points=32768) `。

## 在 BEVFormer 中加入点云数据

### 使用 BEVFormer 加入激光

方法参考 github 代码相关 tag： [v1.0](https://github.com/daxiongpro/UniAD/tree/v1.0), [v2.0](https://github.com/daxiongpro/UniAD/tree/v2.0)

### 使用 BEVFusion 方法

问题1：bevfusion 使用 bev_pool 算子，这个算子在 uniad 使用的 mmdet3d 代码中没有。

解决1：在 bevfusion 代码中，复制 bev_pool 这个算子（整个文件夹），到 mmdet3d 的 ops 中。再复制 setup.py 相关代码。重新编译 mmdet3d 即可。

# 日期

* 2023/12/14：记录 BEVFusion 方法问题
* 2023/11/10：完成历史帧的读取和点云数据的预处理
