# 使用 vscode 调试代码

vscode 是一款代码编辑器，不是 IDE，需要手动配置一些配置文件才能够进行调试。

## 调试 python 代码

编辑项目根目录下的 `.vscode/launch.json` 文件，如果没有这个文件，就自己创建一个。

```json
{
  "configurations": [
    {
      "name": "Python: Current File",
      "type": "python",
      "request": "launch",
      "program": "${file}",
      "console": "integratedTerminal",
      "justMyCode": true
    },
    {
      "name": "train",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/train.py",
      "args": [
        "args1",
        "--args2_name",
        "args2"
      ],
      "console": "integratedTerminal",
      "justMyCode": false //可以调试到环境中其他库的代码
    },
    {
      "name": "debugpy",
      "type": "python",
      "request": "attach",
      "connect": {
        "host": "127.0.0.1",
        "port": 8531
      },
      "justMyCode": false //可以调试到环境中其他库的代码
      // python -m debugpy --listen 8531 --wait-for-client args1 args2 ...
    }
  ]
}

```

> 这里使用了3种方式：
>
> * 运行当前文件：Python: Current File
> * 运行指定文件：（例如） train
>
> * 使用 debugpy 运行：debugpy

保存之后，点击“运行和调试 (ctrl+shift+D)"，就可以根据 name 字段选择。这里说说如何使用 debugpy 调试。

### debugpy 调试 python 代码

在训练 bevfusion 代码时，作者使用了自己的库 torchpack，无法正常调试。因此使用 debugpy 调试。调试方法如下。

* 在命令行终端运行 python 命令时，在 `python` 和`参数`之间加入 `-m debugpy --listen 8531 --wait-for-client`，如：

  ```
  python -m debugpy --listen 8531 --wait-for-client args1 args2 ...
  ```

  > 此时，程序不会运行下去。
  >
* 设置断点
* vscode 调试方法中选择 debugpy，按下 F5 调试

  > 此时程序执行，并停在断点处。
  >

## 调试 C++ 代码

未完待续……

# 日期

2023/05/19：创作本文
