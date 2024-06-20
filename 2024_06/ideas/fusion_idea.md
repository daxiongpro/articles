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

## 方法

F-Pointnet 的问题：

2D 检测的 frustum 内的点作为前景点，如果目标物体被遮挡一部分，点云也会扫到，会有噪声
