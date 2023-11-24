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

### 编码器总体结构

在多视图图像特征中加入 Lidar 点云模态的数据可以充分利用两种模态的优势。在本项工作中，我们给出了一个基于时序 BEV 范式的多模态融合框架。总体框架结构如图 X 所示。其中，多视图图像和 LiDAR 点分别输入到两个单独的主干网络中以提取多模态的特征。然后将 3D 位置与这些特征一起编码成一个坐标编码 encoding，得到两种模态的 tokens。受到 BEVFormer 的启发，由于车辆在行驶的过程中，物体的位置变化不大，下一帧的车辆很可能在上一帧的附近。因此，上一帧的 BEV 特征，在下一帧具有非常重要的参考意义。在此，我们延用 BEVFormer 的时序架构，在历史帧中，同样导入 Lidar 点云数据。

### Position-guided BEV Queries Generator

首先预定义一些 BEV 网格形状的 Query。Query 的长度为 n，其大小为 BEV 网格的形状，即 n = H * W。具体来说，在某个位置 p(x, y) 的 Query 只查询在 BEV 空间中对应一小部分区域。每一个小网格的长和宽，在真实世界的3D 空间中对应 s 米，并以自车坐标为 BEV 特征的原点。并且，在进行 BEV 查询工作之前，对 Query 加入可学习的 PE 模块(Positional embedding)，来引导其查询的位置。我们首先在 3D BEV 空间中，预生成 n 个 anchor 点 (xi, yi, zi)，i 属于 [1, n]。这 n 个点范围统一到 [0, 1] 之间的随机值。然后将这 n 个点转换到 3D 世界坐标系下。其推导如下：

因为 (xi-0)/(1-0) = (xi'-xmin)(xmax - xmin)，所以得到 xi'=xi*(xmax-xmin)+xmin。yi 和 zi 同理。上述公式中，xmax xmin 分别代表在 3D 世界坐标系下的 RoI(region of interest)。


### CE for Point Clouds.

### 多模态 Spatial Cross-Attention.

# 日期

2023/11/10：完成历史帧的读取和点云数据的预处理
