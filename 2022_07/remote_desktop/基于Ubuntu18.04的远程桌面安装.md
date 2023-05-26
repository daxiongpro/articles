# Install remote desktop on Ubuntu 18.04

## 在 `电脑1`**上安装远程桌面，并用** `电脑2`远程登录电脑1的用户

## Install

```bash
wget -c http://c-nergy.be/downloads/xRDP/xrdp-installer-1.2.zip
unzip xrdp-installer-1.2.zip
chmod +x xrdp-installer-1.2.sh
./xrdp-installer-1.2.*
# reboot
```

## Note:

* 重启后不要在 `电脑1`登录，直接使用 `电脑2`登录

## Questions

### 闪退

* 说明在 `电脑1`上登录了，此时重启，然后直接用 `电脑2`登录。

### 黑屏或者花屏

* 打开startwm.sh文件

```bash
sudo vim /etc/xrdp/startwm.sh
```

* 在test -x /etc/X11/Xsession && exec /etc/X11/Xsession前添加如下代码

```bash
gnome-session
. /etc/X11/Xsession
```

* 重启xrdp服务

```bash
sudo systemctl restart xrdp
```

### 参考文献：

[Ubuntu18.04.4 安装XRDP远程桌面终极、最简方案](https://blog.csdn.net/weixin_43315707/article/details/107518380?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-1-107518380-blog-124164589.pc_relevant_paycolumn_v3&spm=1001.2101.3001.4242.2&utm_relevant_index=3)

[使用 xrdp 远程登录ubuntu出现黑屏或者花屏](https://blog.csdn.net/Fatmear/article/details/122037566)
