# 吉利云服务器(PAI 平台)的使用

## 1.总体流程：

1.1.制作 docker 镜像，打包上传到吉利 harbor

方法1：本地电脑使用 docker 配置环境，跑通代码，保存镜像，上传到 docker hub。然后通过 VDI 拉取 docker image，再上传到吉利 harbor 仓库。

方法2：本地电脑使用 docker 配置环境，跑通代码，保存镜像，上传到吉利 harbor 仓库。

1.2.PAI 平台 DSW 中使用 harbor 中上传的 docker 镜像。

## 2.申请 VDI

打开 BPM 提流程 "信息系统账号及权限申请"。里面的“模块"填写：乌班图虚拟工作站。

审批时间：1天。审批完成后，会发邮件到吉利邮箱，内置 VDI 使用说明。

## 3.VDI 里面下载镜像

VDI 内置 nvidia-smi 和 docker。可以从吉利 harbor 或者 docker hub 上 pull image。

### 3.1.从 docker hub 上 pull image

在 docker hub 的用户名之前，加上 pkg.geely.com/docker，就能拉取 docker hub 上的镜像了。

例如：想拉取 docker hub 上 daxiongpro/qdot2004nvidia 这个仓库，docker hub 上的命令为：

```bash
docker pull daxiongpro/qdot2004nvidia:zsh
```

在实际拉取的时候，命令应该为：

```bash
docker pull pkg.geely.com/docker/daxiongpro/qdot2004nvidia:zsh
```

如果遇到问题，就过一段时间再拉取，或者多拉几次。

为完待续。。。

## 日期

2023/07/14：文章撰写
