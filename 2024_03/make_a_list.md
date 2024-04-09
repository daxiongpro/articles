# 打榜任务分配

适配 waymo 数据集、TTA 调研测试、加旷视数据集训练。

适配 waymo 数据集

* waymo 数据集下载：夏勇、张志尧、张书瑞(看哪个 tar 包下载不成功就下哪个上传)
* waymo 数据集上传到 PAI 平台并成功解压：张志尧
* 适配 waymo 数据集——deepInteraction：高佳誉（生成 pkl、编写数据集、eval，训练）
* 适配 waymo 数据集——bevfusion：蔡正奕
* 适配 waymo 数据集——cmt：夏勇
* waymo 数据集测试结果放到 waymo 服务器上测试流程跑通：夏勇、张书瑞(张书瑞拿一个已经有的结果给夏勇，让夏勇放到 waymo 官网上)
* 训练 deepInteraction、bevfusion、cmt

TTA 调研测试：夏勇

加旷视数据集训练：蔡正奕

论文方法调研：蔡正奕

| 方法             | NDS   | 开源 | TTA |
| ---------------- | ----- | ---- | --- |
| EA-LSS           | 0.776 | 1    | 1   |
| CMT              | 0.770 | 1    | 1   |
| DeepIntereaction | 0.763 | 1    | 1   |
| MSMDFusion       | 0.751 | 1    | 1   |
| DAL              | 0.748 | 1    | 0   |
| FocalFormer3D-F  | 0.745 | 1    | 0   |
| UniTR            | 0.745 | 1    | 0   |
