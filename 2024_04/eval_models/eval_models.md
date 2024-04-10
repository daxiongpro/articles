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
* model：dak-base.py
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

## 日期

* 2024/04/10：dal-base、dal-large 测试结果
