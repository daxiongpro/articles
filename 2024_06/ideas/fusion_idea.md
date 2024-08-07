# 多模态融合

## 摘要

思考一下 3D 标注的过程：

易检测的目标直接在点云上标注。

难检测的目标(远处或小目标)：

* 图像视角发现目标
* 去点云对应位置找扫描到的点
* 在点云上绘制 bbox

其实易检测的目标也可以按照难检测的目标检测过程来做。

## 相关工作

### 纯电云检测

#### 稠密范式(BEV)

TODO

#### 稀疏范式

* [FSD](https://github.com/tusen-ai/SST) (Fully Sparse Fusion for 3D Object Detection)。首先全景分割，再把前景点投票

### 融合检测

* F-Pointnet
* [FSF](https://github.com/BraveGroup/FullySparseFusion) (FullySparseFusion)

### FSD 问题

问题一：全景分割慢，可以由 2D 检测的 frustum 内的点作为前景点。(想法已经有人做了：FSF)

全景分割是每个点分类，需要额外的分割网络。有成熟的 2D 目标检测器，可以通过 img 发现目标，得到 img 上的 bbox。再通过视锥得到前景点云。

问题二：前景点投票

只需要在 bev 视角下投票。即原始点 xyz，投票到 bev 下为 X,Y，Z 轴高度方向不需要投票。然后在 BEV 空间下聚类，不需要在 3D 空间下聚类。

### F-Pointnet 的问题

2D 检测的 frustum 内的点作为前景点，如果目标物体被遮挡一部分，点云也会扫到，会有噪声。

## 方法

图像 2D 分割所有 object。对于所有的激光点云 points，取出落在 object 内的 points。然后有两种做法：

* 方法一：激光只用来测距，其他的分类、2D 回归都由 2D 分割任务完成。2D 回归包括 ([x], y, z, l, w, h, yaw) 中除了深度 x 值。
* 方法二：直接使用激光回归。直接使用 FSD 算法回归。也就是 FSD 算法中的前景背景分割，由 2D mask 完成。(思想与 FSF 相似)

但是在 FSF 论文中提到，落在 object 内的 points 也会包含少量背景杂点，这些点是噪音。

### 分割前景点

* 获取由 2D box 生成的视锥中的点云
* 点云投影到图像上，判断每个点是否在 2d mask 内部。记录下所有在内部的点的 index，然后从原始点云去取这些 index。
