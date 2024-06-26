# 记录各种模型测试结果

本文记录不同算法在公开数据集的测试结果。

## CMT

配置参数：

* 数据集：nuscnes-mini
* config：cmt_voxel0075_vov_1600x640_cbgs.py
* pth 来源：官方
* 测试 gpu：xxx

结果：NDS=0.5913

![1712731643100](image/eval_models/nuscenes_mini_cmt_vov.png)

配置参数：

* 数据集：nuscnes-big
* config：cmt_voxel0075_vov_1600x640_cbgs.py
* pth 来源：官方
* 测试 gpu：1*3090

结果：NDS=0.7289

![1712731834193](image/eval_models/nuscenes_big_cmt_vov.png)

## DAL

配置参数：

* 数据集：nuscnes-mini
* model：dal-base.py
* pth 来源：官方
* 测试 gpu：1*3090

结果：NDS=0.6008

![1712731379568](image/eval_models/nuscnes_mini_dal_base.png)

配置参数：

* 数据集：nuscnes-big
* model：dal-base.py
* pth 来源：官方
* 测试 gpu：8*A100

结果：NDS=0.7346

![1712731226518](image/eval_models/nuscnes_big_dal_base.png)

配置参数：

* 数据集：nuscnes-big
* model：dal-large.py
* pth 来源：官方
* 测试 gpu：8*A100

结果：NDS=0.7396

![1712733954780](image/eval_models/nuscnes_big_dal_large.png)

配置参数：

* 数据集：nuscnes-big
* model：dal-large.py
* pth 来源：官方的 dal-large 训练 20 个 epoch 得到(8*A100)
* 测试 gpu：8*A100

结果：NDS=0.7453 (比官方权重效果好，惊喜)

![1713145171421](image/eval_models/nuscenes_big_dal_large_epoch20.png)

配置参数：

* 数据集：nuscnes-big
* model：dal-large.py
* pth 来源：官方的 dal-large 训练 19 个 epoch 得到(8*A100)
* 测试 gpu：8*A100

结果：NDS=0.7459 (跑 19 个 epoch 比 20 个 epoch 效果好)

![1713147961488](image/eval_models/nuscenes_big_dal_large_epoch19.png)

## DeepInteraction

配置参数：

* 数据集：nuscnes-big
* model：Fusion_0075_refactor.py
* pth 来源：官方
* 测试 gpu：8*A100

结果：NDS=0.6909

![1713271853950](image/eval_models/nuscenes_big_deepinteraction_base.png)

配置参数：

* 数据集：nuscnes-big
* model：Fusion_0075_refactor.py
* pth 来源：官方 Fusion_0075_refactor.py 训练 6 个 epoch 得到(8*A100)
* 测试 gpu：8*A100

结果：NDS=0.7248

![1713402633204](image/eval_models/nuscenes_big_deepinteraction_base_epoch6.png)

## 日期

* 2024/04/18：DeepInteraction_base 按照官方训练 6 个 epoch 后测试结果
* 2024/04/16：DeepInteraction_base 官方测试结果
* 2024/04/15：dal-large 按照官方训练 20 个 epoch 后测试结果
* 2024/04/10：dal-base、dal-large 测试结果
