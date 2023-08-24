# 计算机视觉经典论文

## R-CNN

Q：R-CNN 介绍？

A：input -> proposals -> cnn feature extract -> fully connected layer -> SVM， 具体步骤如下：

* 输入图片。
* 利用选择性搜索 (selective search) 算法提取所有 proposals（大约2000幅images。
* 调整（resize/warp）proposals 成固定大小，以满足 CNN输入要求（因为全连接层的限制）。
* 选择一个预训练 （pre-trained）神经网络（如AlexNet、VGG），重新训练全连接层，然后将 feature map 保存到本地磁盘。
* 训练SVM。利用 feature map 训练SVM来对目标和背景进行分类（每个类一个二进制SVM。
* 边界框回归（Bounding boxes Regression）。训练将输出一些校正因子的线性回归分类器。

Q：R-CNN有哪些创新点？

A：使用CNN（ConvNet）计算 feature vectors。从经验驱动特征（SIFT、HOG）到数据驱动特征（CNN feature map），提高特征对样本的表示能力。

## Fast R-CNN

Q：Fast R-CNN 改进了哪些？

A：如下：

* 只对整幅图像进行一次特征提取，避免 R-CNN 中的冗余特征提取。然后将 proposals 投影到 feature map 上。
* 用 RoI pooling 层替换最后一层的 max pooling 层，使得将不同大小的 ROI 转换为固定大小。
* Fast R-CNN网络末尾采用并行的不同的全连接层，可同时输出分类结果和窗口回归结果。

## Faster R-CNN

Q：Faster R-CNN 改进了哪些？

A：使用候选区域网络（RPN）代替 selective search。

Q：介绍一下 RPN 网络？

A：就是一个 3x3 的卷积网络，最后用全连接层，接上分类头(2k)和回归头(4k)，k代表 region proposals 的个数。分类头就判断是否是前景(有 object )；回归头用于回归 region proposals。RPN 是对于特征图中的每一个位置(像素)，做k次预测。回归的时候，是回归相对于锚框的偏移量。

## FPN

Q：FPN 解决了什么问题？

A：卷积神经网络中，浅层 feature map 分辨率高，但是语义特征不明显；深层 feature map 分辨率低，但是包含更多的语义特征。深层 feature map 分辨率低是因为，传统卷积网络在下采样的过程中，由于池化和卷积的stride，会丢失一些信息(像素点)，导致小物体检测性能下降。因此，特征金字塔网络，在每一层将语义特征和高分辨率特征相融合。其中上采样的方法看[这里](https://zhuanlan.zhihu.com/p/92005927)。

Q：Faster R-CNN 中有用 FPN 吗？

A：原版的没有。但是可以将 FPN 网络用到 RPN 网络中。

## Mask R-CNN

Q：Mask R-CNN 的构成？

A：Mask R-CNN

= Faster R-CNN+FCN

= ResNeXt+RPN+RoI Align+Fast R-CNN+FCN

Q：Mask R-CNN 的创新点有哪些？

A：如下：

* Backbone：ResNeXt-101+FPN
* RoI Align替换RoI Pooling

Q：RoI Pooling 有什么缺点？

A：在常见的两级检测框架（比如Fast-RCNN，Faster-RCNN，RFCN）中，RoI Pooling 的作用是根据预选框的位置坐标在特征图中将相应区域池化为固定尺寸的特征图，以便进行后续的分类和包围框回归操作。由于预选框的位置通常是由模型回归得到的，一般来讲是浮点数，而池化后的特征图要求尺寸固定。故RoI Pooling这一操作存在两次量化的过程。

* 将候选框边界量化为整数点坐标值。
* 将量化后的边界区域平均分割成 k×k 个单元(bin),对每一个单元的边界进行量化。

事实上，经过上述两次量化，此时的候选框已经和最开始回归出来的位置有一定的偏差，这个偏差会影响检测或者分割的准确度。

Q：RoI Align 怎么做？
