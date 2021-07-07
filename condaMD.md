# anaconda 问题

## anaconda 新建虚拟环境 目录问题：

问题描述：

在创建虚拟环境时：`conda create -n test python=3.7` 回车，将环境建到`/home/xxx/.conda/envs/`里面。而我希望把环境建到`/usr/miniconda3/envs/`，即全局miniconda3安装目录里。

解决方法:

`conda create --prefix /usr/miniconda3/envs/`



### 2080Ti 的pointnet2代码移植到3080出问题

#### 问题1：

2080ti上环境为pytorch17，在3080上安装pytorch17+cuda11.0成功，但编译pointnet2的时候出现：

CUDA error: no kernel image is available for execution on the device

##### 原因：

3080的算力有8.6，cuda11.0最高不支持7.几（好像是7.5？）

##### 解决办法：

* 升级pytorch+cuda = pytorch1.8.0+cuda11.1

然后再次编译pointnet2又碰到问题：

#### 问题2：

subprocess.CalledProcessError: Command '['ninja', '-v']' returned non-zero e

##### 解决办法：

* vim编辑 ..../site-packages/torch/utils/cpp_extension.py。
* [‘ninja’, ‘-v’]改为[‘ninja’, ‘–version’]

参考文献：

* [问题1](https://www.codeleading.com/article/92975632597/)
* [问题2](https://blog.csdn.net/weixin_43731803/article/details/116787152https://)
