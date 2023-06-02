# 配置文件——yaml 转 py 格式

在 bevfusion 中，作者使用 yaml 代替 py 文件作为 config 文件，蛋疼。但是可以使用 mmcv 将 yaml 转化为 py。代码如下：

```python
from mmcv import Config

yaml_path = 'xxx/configs.yaml'
py_path = 'xxx/configs.py'
with open(py_path, 'w') as f:  # 如果 py_path 文件不存在，则创建文件
    pass

cfg_yaml = Config.fromfile(yaml_path)  # 读取yaml 文件
cfg = Config(cfg_yaml._cfg_dict, filename=py_path)
cfg.dump(py_path)

```

上述代码中，yaml_path 是读取的 yaml 文件，py_path 是要保存的 py 文件。

# 日期

2023/06/02：文章创作日期
