# tmux 相关

## tmux 设置鼠标滚动

在 Tmux 中启用鼠标滚动需要对 `tmux.conf` 文件进行相应的配置。你可以按照以下步骤进行操作：

1. 打开或创建 `tmux.conf` 文件。你可以在用户的主目录下找到这个文件，如果不存在，可以创建一个。

```bash
vim ~/.tmux.conf
```
2. 在 `tmux.conf` 文件中添加以下配置行：

```bash
set -g mouse on
```

   这将启用鼠标支持。
3. 保存并关闭文件。
4. 重新加载 tmux 配置，可以使用以下命令：

```bash
tmux source-file ~/.tmux.conf
```

现在，你应该能够在 tmux 窗口中使用鼠标滚动功能了。请注意，这可能会因终端和 tmux 版本的不同而有所差异，因此在使用前最好查阅相应版本的文档。

> 答案来自：chatGPT

# 时间

2023/11/23：init
