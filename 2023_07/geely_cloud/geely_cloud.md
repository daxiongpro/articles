# 吉利云服务器(PAI 平台)的使用

连接 PAI 平台有两种方法：

* 任意电脑 -> VDI -> PAI 平台
* 公司电脑 -> PAI 平台

## 1.总体流程：

环境准备

* 制作 docker 镜像，打包上传到吉利 harbor
* PAI 平台 DSW 中新建 docker 容器

代码准备

* 本地调试好代码，打包拷贝到公司电脑
* filezillar 上传到 PAI 平台 DSW

## 2.申请 VDI

打开 BPM 提流程 "信息系统账号及权限申请"。里面的“模块"填写：乌班图虚拟工作站。

审批时间：1天。审批完成后，会发邮件到吉利邮箱，内置 VDI 使用说明。

> 连接 VDI 的时候，如果连接不上，那么就是你的本地电脑的 IP 被公司防火墙挡了，需要申请开通。

开通方法：

* 登陆 itsm.geely.com，选择防火墙申请，填写相关信息。
* 开通负责人是"奕水兴"，有问题企业微信找他。

## 3.镜像上传到 harbor

镜像上传到 harbor 有两种方法：

1.本地直接上传

本地电脑使用 docker 配置环境，跑通代码，保存镜像，上传到吉利 harbor 仓库。

> 连接 harbor 的时候，如果连接不上，同样需要申请开通，方法同上（2.申请 VDI）。

2.通过 VDI 上传

* 本地电脑上传到 docker hub
* VDI 从 docker hub 拉取 docker image
* VDI 上传到吉利 harbor 仓库

### 3.1.VDI 里面下载镜像

VDI 内置 nvidia-smi 和 docker。可以从吉利 harbor 或者 docker hub 上 pull image。想要从 docker hub 上拉取镜像，在 docker hub 的用户名之前，加上 pkg.geely.com/docker，就能拉取 docker hub 上的镜像了。例如：想拉取 docker hub 上 daxiongpro/qdot2004nvidia 这个仓库，docker hub 上的命令为：

```bash
docker pull daxiongpro/qdot2004nvidia:zsh
```

在实际拉取的时候，命令应该为：

```bash
docker pull pkg.geely.com/docker/daxiongpro/qdot2004nvidia:zsh
```

如果遇到问题，就过一段时间再拉取，或者多拉几次。

### 3.2.镜像推送到 harbor

修改 tag，再上传。上传教程看 harbor 网站，或者问"孙责荃"、"郭昀"。

## 4.代码推送到 PAI 平台

步骤：

* 代码打包
* 拷贝到公司电脑
* filezillar 上传到 PAI 平台

## 日期

2023/07/18：更新：开通 VDI 、代码推送

2023/07/14：文章撰写
