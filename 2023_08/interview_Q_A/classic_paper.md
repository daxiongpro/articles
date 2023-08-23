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
