# 使用 python 解析 pcd 文件

pcd 是一种点云文件格式，其包含一些头。要取其中的点云数据，需要采取一些方法。

## 方法1：使用 numpy 直接读取二进制文件

使用 numpy 读取 pcd 的二进制文件，再根据 pcd 文件格式，将点云数据部分取出。

```python
import numpy as np
def read_pcd_bin(pcd_file):
    with open(pcd_file, 'rb') as f:
        data = f.read()
        data_binary = data[data.find(b"DATA binary") + 12:]
        points = np.frombuffer(data_binary, dtype=np.float32).reshape(-1, 3)
        points = points.astype(np.float32)
    return points
xyz = read_pcd_bin("test.pcd")  # (N, 3)
```

## 方法2：使用 pyntcloud 库

```python
from pyntcloud import PyntCloud
points = PyntCloud.from_file("test.pcd")
xyz = points.xyz  # (N, 3)
```

## 日期

2023/5/10：创作本文
