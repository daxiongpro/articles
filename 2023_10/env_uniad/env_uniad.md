# 环境调试——UniAD

## 环境安装

1.在能联网的机器上安装环境，直接按照官方 README.md 就能成功。

2.在不能联网的云服务器上安装环境，应当在本地先把环境基本调试成功，然后将配好的 conda 环境打包，传到云服务器上。在服务器上需要重新手工安装 mmdet3d，但是会遇到以下问题：

### 问题1

Q：安装 mmdet3d 的时候， `no matching distribution found for numpy<1.20.0`

A：先将 numpy 降级成 1.19.5，再安装 1.20.0。都离线用 .whl 文件安装

### 问题2

Q：包没安装

A：需要安装 tomli 2.0.1

### 问题3

Q：no kernel image...

A：看是哪个库报这个错。我的是 mmcv-full，重新下载 whl ，然后上传到云平台，离线安装。

### 问题4

Q：出现：`TypeError: FormatCode() got an unexpected keyword argument ‘verify‘`

A：原来yapf包的0.40.2，降低为yapf==0.40.1，问题就可解决。

### 问题5

Q：在训练的时候出现： `torch.distributed.elastic.multiprocessing.errors.ChildFailedError`

A：其实不是这句报错本身的问题，往前看对应的错误。解决前面的错误，这个报错会消失。

## 运行报错

Q：运行可视化代码 `./tools/uniad_vis_result.sh` 出现

```bash
Traceback (most recent call last):
  File "./tools/analysis_tools/visualize/run.py", line 342, in <module>
    main(args)
  File "./tools/analysis_tools/visualize/run.py", line 304, in main
    viser = Visualizer(version='v1.0-trainval', predroot=args.predroot, dataroot='data/nuscenes', **render_cfg)
  File "./tools/analysis_tools/visualize/run.py", line 66, in __init__
    self.predictions = self._parse_predictions_multitask_pkl(predroot)
  File "./tools/analysis_tools/visualize/run.py", line 115, in _parse_predictions_multitask_pkl
    trajs = outputs[k][f'traj'].numpy()
KeyError: 'traj'
```

A：先运行

```
./tools/uniad_dist_eval.sh ./projects/configs/stage2_e2e/base_e2e.py ./ckpts/uniad_base_e2e.pth 4
```

## 时间

2023/11/09：更改格式更清楚

2023/10/27：环境安装章节
