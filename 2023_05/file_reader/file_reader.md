# python 文件操作

### 读取 pkl, json, yaml 等文件

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

### 文件内容修改并写入

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

# 修改文件
data['a'] = 0

# 文件写入
with open(file_, 'wb') as f:
    if file_.endswith('.pkl'):
        pickle.dump(data, f)
	print('all success !')
    elif file_.endswith('.json'):
        pass
    elif file_.endswith('.yaml'):
        pass

```


## 日期

2023/11/09：更改标题，增加文件修改和写入操作

2023/05/31：文章创作日期
