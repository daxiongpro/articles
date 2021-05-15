

# anaconda 问题

## anaconda 新建虚拟环境 目录问题：

问题描述：

在创建虚拟环境时：`conda create -n test python=3.7` 回车，将环境建到`/home/xxx/.conda/envs/`里面。而我希望把环境建到`/usr/miniconda3/envs/`，即全局miniconda3安装目录里。

解决方法:

`conda create --prefix /usr/miniconda3/envs/`

