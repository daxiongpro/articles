# 方法1
import numpy as np


def read_pcd_bin(pcd_file):
    with open(pcd_file, 'rb') as f:
        data = f.read()
        data_binary = data[data.find(b"DATA binary") + 12:]
        points = np.frombuffer(data_binary, dtype=np.float32).reshape(-1, 3)
        points = points.astype(np.float32)
    return points


xyz = read_pcd_bin("test.pcd")  # (N, 3)


# 方法2
from pyntcloud import PyntCloud
points = PyntCloud.from_file("test.pcd")
xyz = points.xyz  # (N, 3)
