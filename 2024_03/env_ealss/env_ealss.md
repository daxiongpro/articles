# 环境调试——EA-LSS

## 1.环境安装

1.1.复制以下环境到项目根目录，文件名"requirement.txt"。这环境是由[BEVFusion（北大&amp;阿里）环境搭建教程](https://blog.csdn.net/u014295602/article/details/127933607) 的作者导出，我再根据之前的文章[离线安装 python 库](../../2023_11/pkg_install_offline/pkg_install_offline.md)，使用 python 导出。

```bash
# requirement.txt
absl-py==1.3.0
addict==2.4.0
anyio==3.6.2
argon2-cffi==21.3.0
argon2-cffi-bindings==21.2.0
asttokens==2.1.0
attrs==22.1.0
backcall==0.2.0
beautifulsoup4==4.11.1
black==22.10.0
bleach==5.0.1
cachetools==5.2.0
certifi==2022.9.24
cffi==1.15.1
charset-normalizer==2.1.1
click==8.1.3
contourpy==1.0.6
cycler==0.11.0
Cython==0.29.32
debugpy==1.6.3
decorator==5.1.1
defusedxml==0.7.1
# depthwise-conv2d-implicit-gemm==0.0.0
descartes==1.1.0
entrypoints==0.4
exceptiongroup==1.0.4
executing==1.2.0
fastjsonschema==2.16.2
filelock==3.8.0
fire==0.4.0
flake8==5.0.4
fonttools==4.38.0
google-auth==2.14.1
google-auth-oauthlib==0.4.6
grpcio==1.50.0
h5py==3.7.0
huggingface-hub==0.11.0
idna==3.4
imageio==2.22.4
importlib-metadata==5.0.0
importlib-resources==5.10.0
iniconfig==1.1.1
ipykernel==6.17.1
ipython==8.6.0
ipython-genutils==0.2.0
ipywidgets==8.0.2
jedi==0.18.1
Jinja2==3.1.2
joblib==1.2.0
jsonschema==4.17.0
jupyter==1.0.0
jupyter_client==7.4.6
jupyter-console==6.4.4
jupyter_core==5.0.0
jupyter-server==1.23.2
jupyterlab-pygments==0.2.2
jupyterlab-widgets==3.0.3
kiwisolver==1.4.4
llvmlite==0.31.0
loguru==0.6.0
lyft-dataset-sdk==0.0.8
Markdown==3.4.1
MarkupSafe==2.1.1
matplotlib==3.6.2
matplotlib-inline==0.1.6
mccabe==0.7.0
mistune==2.0.4
mmcls==0.24.1
mmcv-full==1.4.0
# mmdet==2.11.0mmdet3d==0.11.0mmpycocotools==12.0.3
msgpack==1.0.4
msgpack-numpy==0.4.8
multimethod==1.9
mypy-extensions==0.4.3
nbclassic==0.4.8
nbclient==0.7.0
nbconvert==7.2.5
nbformat==5.7.0
nest-asyncio==1.5.6
networkx==2.2
ninja==1.11.1
notebook==6.5.2
notebook_shim==0.2.2
numba==0.48.0
numpy==1.23.4
nuscenes-devkit==1.1.9
oauthlib==3.2.2
opencv-python==4.6.0.66
packaging==21.3
pandas==1.4.4
pandocfilters==1.5.0
parso==0.8.3
pathspec==0.10.2
pexpect==4.8.0
pickleshare==0.7.5
Pillow==9.3.0
pip==22.3.1
pkgutil_resolve_name==1.3.10
platformdirs==2.5.4
plotly==5.11.0
pluggy==1.0.0
plyfile==0.7.4
prettytable==3.5.0
prometheus-client==0.15.0
prompt-toolkit==3.0.32
protobuf==3.20.3
psutil==5.9.4
ptyprocess==0.7.0
pure-eval==0.2.2
pyasn1==0.4.8
pyasn1-modules==0.2.8
pycocotools==2.0.6
pycodestyle==2.9.1
pycparser==2.21
pyflakes==2.5.0
Pygments==2.13.0
pyparsing==3.0.9
pyquaternion==0.9.9
pyrsistent==0.19.2
pytest==7.2.0
python-dateutil==2.8.2
pytz==2022.6
PyWavelets==1.4.1
PyYAML==6.0
pyzmq==24.0.1
qtconsole==5.4.0
QtPy==2.3.0
requests==2.28.1
requests-oauthlib==1.3.1
rsa==4.9
scikit-image==0.19.3
scikit-learn==1.1.3
scipy==1.9.3
Send2Trash==1.8.0
setuptools==65.5.1
Shapely==1.8.5.post1
six==1.16.0
sniffio==1.3.0
soupsieve==2.3.2.post1
stack-data==0.6.1
tabulate==0.9.0
tenacity==8.1.0
tensorboard==2.11.0
tensorboard-data-server==0.6.1
tensorboard-plugin-wit==1.8.1
tensorpack==0.11
termcolor==2.1.0
terminado==0.17.0
terminaltables==3.1.10
threadpoolctl==3.1.0
tifffile==2022.10.10
timm==0.6.11
tinycss2==1.2.1
toml==0.10.2
tomli==2.0.1
torch==1.8.0+cu111
torchaudio==0.8.0
torchpack==0.3.1
torchvision==0.9.0+cu111
tornado==6.2
tqdm==4.64.1
traitlets==5.5.0
trimesh==2.35.39
typing_extensions==4.4.0
urllib3==1.26.12
wcwidth==0.2.5
webencodings==0.5.1
websocket-client==1.4.2
Werkzeug==2.2.2
wheel==0.38.4
widgetsnbextension==4.0.3
yapf==0.32.0
zipp==3.10.0

```

1.2.安装上述包

```
pip install -r requirement.txt
```

1.3.安装对应版本的 `mmcv==1.4.0`。这里需要手动到官网下载 whl 再安装。参考[BEVFusion（北大&amp;阿里）环境搭建教程](https://blog.csdn.net/u014295602/article/details/127933607)。

1.4.安装 mmdet 和 mmdet3d。都从 EA-LSS 代码中使用 `python setup.py develop` 安装。

## 2.数据预处理

将 nuscenes 数据预处理成 pkl 格式：

```bash
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```

但是会报错：找不到 can_bus_root_path 变量。

原因：EA-LSS 作者在 BEVFuison 上改了代码，但是有 bug。

解决方法：

1.从 nuscenes 官网下载 can_bus 信息，添加到 `data/nuscenes` 目录下，完成之后 `data/` 目录结构为：

```bash
nuscenes/
├── can_bus
├── maps
├── samples
├── sweeps
└── v1.0-trainval
```

2.修改 `create_data.py`，将

```python
    nuscenes_converter.create_nuscenes_infos(
        root_path, info_prefix, version=version, max_sweeps=max_sweeps)
```

改为：

```python
    nuscenes_converter.create_nuscenes_infos(
        root_path, root_path, info_prefix, version=version, max_sweeps=max_sweeps)
```

其中，`root_path` 指向 `data/nuscenes/`。

## 3.模型训练

按照 [EA-LSS 官方 README](https://github.com/hht1996ok/EA-LSS)。

## 参考资料

* [BEVFusion（北大&amp;阿里）环境搭建教程](https://blog.csdn.net/u014295602/article/details/127933607)
