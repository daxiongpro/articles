## 论文解读——PERT

DETR3D 的问题：

* 参考点投影机制：如果参考点出错，采样到的图像特征是无效的。
* 单点特征采样：参考点特征是 local ，不够 global。
* 采样方法比较复杂，petr 生成 3d 感知特征，简化投影过程。

![1694056241100](image/PETR_paper/petr_1.png)

> 上图可以看出，PETR 与 DETR 结构类似，不需要参考框，以及反向投影到 2D 图像的操作。其中的关键就是 3D PE，即 3D positional encode。下面介绍如何得到 3D PE。

### 整体流程

![1694056241100](image/PETR_paper/petr_2.png)

> 上图中，分为上下两个分支。上面的分支为 backbone，对图像进行特征提取，下方虚线框内的就是 3D PE，其目的是为了产生 2D feature map 每个像素上对应的 3D 特征。其中包含两个步骤：1. 3D 坐标生成; 2. 3D positional encoder。然后将 2D feaure map 和 3D 位置编码同时输入到 3D positional encoder 模块。

### 详细过程

#### 3D 坐标生成

![1694056241100](image/PETR_paper/petr_3.png)

> 上图表示 3D 坐标生成过程。2D 到 3D 坐标转换可以通过相机参数算出来。首先预设 d 个深度值，feature map 上的点 (u, v) 和每个深度值共同构建了图像坐标系的点 (u, v, d)。然后通过相机内外参将 (u, v, d)转换到 3D 空间坐标，然后采用位置编码操作进 3D 位置编码。

#### 3D positional encoder

![1694056241100](image/PETR_paper/petr_4.png)

> 上图为 3D positional encoder 模块，说明了 3D 位置编码生成过程：2D 图像上的每个像素都有 D 个深度，每个深度的位置都有一个 3D 坐标 xyz，即图中的 D x 4。4 表示 xyz1。多一个 1 是因为在矩阵计算时方便，没有实际含义。经过全连接层 FC 之后，与 2D 图像特征相加。

### 3D positional encoder 的通俗理解

上图黄色矩形为 3D positional encoder 模块。其形状为(N, H, W, D*4)。其中，N 表示有 N 个视角的相机，在 nuscenes 中为 6。对于每一个视角，是一个 feature map 大小的 tensor，这个 tensor 包含了 3D 的位置编码，位置编码为 D * 4。也就是说，feature map 上的每一个 pixel，除了包含语义信息外，还都蕴含了 D 个深度坐标的 3D 位置信息。

### 参考资料

* [b 站视频讲解——自动驾驶之心](https://www.bilibili.com/video/BV1ru4y1v7PY/?spm_id_from=333.337.search-card.all.click)

### 时间

* 2023/9/13——文章更新：增加 3D positional encoder 的通俗理解
* 2023/9/11——文章撰写日期
