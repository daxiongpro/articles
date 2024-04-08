# 离线安装 python 库

在一些服务器上，我们搭建完Python环境之后，因为服务器的网络限制原因，不能直接通过pip命令下载安装Python的依赖包。

本文我们以安装 spconv-cu111 库为例，介绍如何在服务器离线安装 python 库。介绍两种方法，推荐方法二。

## 方法一

### 1.查看依赖库并导出

1.1.查看 spconv-cu111 的依赖库以及版本，使用以下命令：

```bash
pipdeptree -p spconv-cu111
```

输出：

```bash
Warning!!! Possibly conflicting dependencies found:
* catkin-pkg==0.5.2
 - docutils [required: Any, installed: ?]
------------------------------------------------------------------------
spconv-cu111==2.1.21
├── cumm-cu111 [required: >=0.2.8, installed: 0.2.9]
│   ├── ccimport [required: <0.4.0, installed: 0.3.7]
│   │   ├── ninja [required: Any, installed: 1.11.1]
│   │   ├── pybind11 [required: Any, installed: 2.11.1]
│   │   └── requests [required: Any, installed: 2.31.0]
│   │       ├── certifi [required: >=2017.4.17, installed: 2023.7.22]
│   │       ├── charset-normalizer [required: >=2,<4, installed: 3.2.0]
│   │       ├── idna [required: >=2.5,<4, installed: 3.4]
│   │       └── urllib3 [required: >=1.21.1,<3, installed: 1.26.16]
│   ├── fire [required: Any, installed: 0.5.0]
│   │   ├── six [required: Any, installed: 1.16.0]
│   │   └── termcolor [required: Any, installed: 2.3.0]
│   ├── numpy [required: Any, installed: 1.22.4]
│   ├── pccm [required: <0.4.0, installed: 0.3.4]
│   │   ├── ccimport [required: >=0.3.1, installed: 0.3.7]
│   │   │   ├── ninja [required: Any, installed: 1.11.1]
│   │   │   ├── pybind11 [required: Any, installed: 2.11.1]
│   │   │   └── requests [required: Any, installed: 2.31.0]
│   │   │       ├── certifi [required: >=2017.4.17, installed: 2023.7.22]
│   │   │       ├── charset-normalizer [required: >=2,<4, installed: 3.2.0]
│   │   │       ├── idna [required: >=2.5,<4, installed: 3.4]
│   │   │       └── urllib3 [required: >=1.21.1,<3, installed: 1.26.16]
│   │   ├── fire [required: Any, installed: 0.5.0]
│   │   │   ├── six [required: Any, installed: 1.16.0]
│   │   │   └── termcolor [required: Any, installed: 2.3.0]
│   │   ├── lark [required: >=1.0.0, installed: 1.1.7]
│   │   ├── portalocker [required: >=2.3.2, installed: 2.7.0]
│   │   └── pybind11 [required: >=2.6.0, installed: 2.11.1]
│   └── pybind11 [required: >=2.6.0, installed: 2.11.1]
├── fire [required: Any, installed: 0.5.0]
│   ├── six [required: Any, installed: 1.16.0]
│   └── termcolor [required: Any, installed: 2.3.0]
├── numpy [required: Any, installed: 1.22.4]
├── pccm [required: >=0.2.21, installed: 0.3.4]
│   ├── ccimport [required: >=0.3.1, installed: 0.3.7]
│   │   ├── ninja [required: Any, installed: 1.11.1]
│   │   ├── pybind11 [required: Any, installed: 2.11.1]
│   │   └── requests [required: Any, installed: 2.31.0]
│   │       ├── certifi [required: >=2017.4.17, installed: 2023.7.22]
│   │       ├── charset-normalizer [required: >=2,<4, installed: 3.2.0]
│   │       ├── idna [required: >=2.5,<4, installed: 3.4]
│   │       └── urllib3 [required: >=1.21.1,<3, installed: 1.26.16]
│   ├── fire [required: Any, installed: 0.5.0]
│   │   ├── six [required: Any, installed: 1.16.0]
│   │   └── termcolor [required: Any, installed: 2.3.0]
│   ├── lark [required: >=1.0.0, installed: 1.1.7]
│   ├── portalocker [required: >=2.3.2, installed: 2.7.0]
│   └── pybind11 [required: >=2.6.0, installed: 2.11.1]
└── pybind11 [required: >=2.6.0, installed: 2.11.1]
```

1.2.导出以上输出，使用以下命令：

```bash
pipdeptree --warn silence --warn silence --warn silence -p spconv-cu111 > pipdeptree_out.txt
```

此时会生成一个文件 pipdeptree_out.txt ：

```bash
spconv-cu111==2.1.21
├── cumm-cu111 [required: >=0.2.8, installed: 0.2.9]
│   ├── ccimport [required: <0.4.0, installed: 0.3.7]
│   │   ├── ninja [required: Any, installed: 1.11.1]
│   │   ├── pybind11 [required: Any, installed: 2.11.1]
│   │   └── requests [required: Any, installed: 2.31.0]
│   │       ├── certifi [required: >=2017.4.17, installed: 2023.7.22]
│   │       ├── charset-normalizer [required: >=2,<4, installed: 3.2.0]
│   │       ├── idna [required: >=2.5,<4, installed: 3.4]
│   │       └── urllib3 [required: >=1.21.1,<3, installed: 1.26.16]
│   ├── fire [required: Any, installed: 0.5.0]
│   │   ├── six [required: Any, installed: 1.16.0]
│   │   └── termcolor [required: Any, installed: 2.3.0]
│   ├── numpy [required: Any, installed: 1.22.4]
│   ├── pccm [required: <0.4.0, installed: 0.3.4]
│   │   ├── ccimport [required: >=0.3.1, installed: 0.3.7]
│   │   │   ├── ninja [required: Any, installed: 1.11.1]
│   │   │   ├── pybind11 [required: Any, installed: 2.11.1]
│   │   │   └── requests [required: Any, installed: 2.31.0]
│   │   │       ├── certifi [required: >=2017.4.17, installed: 2023.7.22]
│   │   │       ├── charset-normalizer [required: >=2,<4, installed: 3.2.0]
│   │   │       ├── idna [required: >=2.5,<4, installed: 3.4]
│   │   │       └── urllib3 [required: >=1.21.1,<3, installed: 1.26.16]
│   │   ├── fire [required: Any, installed: 0.5.0]
│   │   │   ├── six [required: Any, installed: 1.16.0]
│   │   │   └── termcolor [required: Any, installed: 2.3.0]
│   │   ├── lark [required: >=1.0.0, installed: 1.1.7]
│   │   ├── portalocker [required: >=2.3.2, installed: 2.7.0]
│   │   └── pybind11 [required: >=2.6.0, installed: 2.11.1]
│   └── pybind11 [required: >=2.6.0, installed: 2.11.1]
├── fire [required: Any, installed: 0.5.0]
│   ├── six [required: Any, installed: 1.16.0]
│   └── termcolor [required: Any, installed: 2.3.0]
├── numpy [required: Any, installed: 1.22.4]
├── pccm [required: >=0.2.21, installed: 0.3.4]
│   ├── ccimport [required: >=0.3.1, installed: 0.3.7]
│   │   ├── ninja [required: Any, installed: 1.11.1]
│   │   ├── pybind11 [required: Any, installed: 2.11.1]
│   │   └── requests [required: Any, installed: 2.31.0]
│   │       ├── certifi [required: >=2017.4.17, installed: 2023.7.22]
│   │       ├── charset-normalizer [required: >=2,<4, installed: 3.2.0]
│   │       ├── idna [required: >=2.5,<4, installed: 3.4]
│   │       └── urllib3 [required: >=1.21.1,<3, installed: 1.26.16]
│   ├── fire [required: Any, installed: 0.5.0]
│   │   ├── six [required: Any, installed: 1.16.0]
│   │   └── termcolor [required: Any, installed: 2.3.0]
│   ├── lark [required: >=1.0.0, installed: 1.1.7]
│   ├── portalocker [required: >=2.3.2, installed: 2.7.0]
│   └── pybind11 [required: >=2.6.0, installed: 2.11.1]
└── pybind11 [required: >=2.6.0, installed: 2.11.1]
```

### 2.修改 requirements.txt 格式

使用以下 python 脚本：

```python
def parse_requirements(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    packages = []
    packages.append(lines[0].strip())
    for line in lines[1:]:
        package_name = line.split("── ")[1].split(" [required: ")[0]
        installed_version = line.split("installed: ")[1].split("]")[0]
        packages.append(f"{package_name}=={installed_version}")

    return packages


if __name__ == "__main__":
    file_path = "/path/to/your/input/dir/pipdeptree_out.txt"
    reuqirements_path = "/path/to/your/output/dir/requirements.txt"
    result = parse_requirements(file_path)
    print(result)
    with open(reuqirements_path, "w") as file:
        for line in result:
            file.write(line + "\n")
    print('保存成功！')
  

```

此时，得到的 requirements.txt 结果为一个标准的 python requirements：

```
spconv-cu111==2.1.21
cumm-cu111==0.2.9
ccimport==0.3.7
ninja==1.11.1
pybind11==2.11.1
requests==2.31.0
certifi==2023.7.22
charset-normalizer==3.2.0
idna==3.4
urllib3==1.26.16
fire==0.5.0
six==1.16.0
termcolor==2.3.0
numpy==1.22.4
pccm==0.3.4
ccimport==0.3.7
ninja==1.11.1
pybind11==2.11.1
requests==2.31.0
certifi==2023.7.22
charset-normalizer==3.2.0
idna==3.4
urllib3==1.26.16
fire==0.5.0
six==1.16.0
termcolor==2.3.0
lark==1.1.7
portalocker==2.7.0
pybind11==2.11.1
pybind11==2.11.1
fire==0.5.0
six==1.16.0
termcolor==2.3.0
numpy==1.22.4
pccm==0.3.4
ccimport==0.3.7
ninja==1.11.1
pybind11==2.11.1
requests==2.31.0
certifi==2023.7.22
charset-normalizer==3.2.0
idna==3.4
urllib3==1.26.16
fire==0.5.0
six==1.16.0
termcolor==2.3.0
lark==1.1.7
portalocker==2.7.0
pybind11==2.11.1
pybind11==2.11.1
```

当前的目录结构为：

```bash
/path/to/your/pkgs
├── pipdeptree_out.txt
└── requirements.txt
```

### 3.下载依赖包

在当前目录(/path/to/your/pkgs)下：

```bash
pip download -r requirements.txt -d packages/ -i https://pypi.tuna.tsinghua.edu.cn/simple
```

上述命令会下载所有包的 whl 文件。

此时的目录结构为：

```bash
/path/to/your/pkgs
├── packages
├── pipdeptree_out.txt
└── requirements.txt
```

### 4.安装依赖包

将当前目录(/path/to/your/pkgs)打包上传到云端，并解压缩，进入当前目录。运行：

```bash
pip install --no-index --find-links=./packages -r ./requirements.txt
```

安装成功！

> 参考资料：[Python之离线安装第三方库（依赖包）](https://blog.csdn.net/bilibalasha/article/details/129155752)

## 方法二

直接下载这个包以及依赖包：

```bash
pip download pip install waymo-open-dataset-tf-2-4-0==1.4.1 -d .
```

上述代码下载了 waymo-open-dataset 以及其依赖包到当前文件夹，然后上传到服务器，进入该目录并安装：

```
pip install ./*
```

# 日期

* 2024/04/08：更新方法二
* 2023/11/16：文章撰写日期
