# python 读取 pkl, json, yaml 等文件

```python
import pickle
import json
import yaml

file_ = 'xx/xxx.pkl'
file_ = "xx/xxx.json"
file_ = 'xx/xxx.yaml'

with open(file_, 'rb') as f:
    if file_.endswith('.pkl'):
        data = pickle.load(f)
    elif file_.endswith('.json'):
        data = json.load(f)
    elif file_.endswith('.yaml'):
        data = yaml.safe_load(f)

print(data)
```

## 日期

2023/05/31：文章创作日期
