### 编写函数三个引号调不出参数模板

- 去File | Settings | Tools | Python Integrated Tools | Docstring format 
- 这里改成你想要的格式，然后再回去看看你的三个引号。默认的可能是plain也就是空的

### 解决git无法clone提示443以及配置git代理方法
原因：pycharm设置了代理，但git没有设置代理

解决办法：
- 打开cmder
- git config --global http.proxy "localhost:1080"