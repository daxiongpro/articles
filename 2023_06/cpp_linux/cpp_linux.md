# Linux 下 c++程序的安装

本文首先动手写一个 Linux 系统下，C++ 动态链接库的 demo，然后介绍如何离线安装 OpenMPI 软件。

## 动态链接库 demo

我们开始动手做一个 C++ 动态连接库，取名为 mylibrary。然后，将 mylibrary 这个库，安装到系统中，并在另一个项目 my_demo_proj 中使用这个库。本章节源码参考 [C++ 极简 demo](https://github.com/daxiongpro/demo_cpp) 。

### 1.原理

* 自己编写的库的源文件首先编译成动态链接库 .so，动态链接库放到系统目录中。
* 当使用这个库的时候，会自动从系统中寻找这个 .so 文件。
* 系统如何寻找这个 .so 文件？根据环境变量：`$LD_LIBRARY_PATH`。

### 2.编写 mylibrary 库并编译安装

过程分为3步骤：

* 源码编写：编写头文件、源文件
* 编译：使用 cmake 或者 g++，将源码编译为 .so 文件
* 安装：将 .so 文件拷贝到指定目录下

#### 2.1.源码编写

* 头文件：mylibrary.h

```cpp
#ifndef MYLIBRARY_H
#define MYLIBRARY_H

void myLibraryFunction();

#endif
```

* 源文件：mylibrary.cpp

```cpp
#include <iostream>

void myLibraryFunction() {
    std::cout << "This is my library function!" << std::endl;
}
```

上述代码中，头文件定义了函数，源文件实现了函数，函数体打印一句话。将上述两个文件放在 mylibrary 文件夹下，就可以开始编译了。

#### 2.2.编译

编译有两种方式：使用 cmake 和使用 g++。

**方法一：使用 cmake**

cmake 文件：CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.5)

# 设置项目名称
project(mylibrary)

# 指定要编译的源代码文件
set(SOURCES
    mylibrary.cpp
)

# 指定要生成的共享库文件名及类型
set(LIBRARY_NAME mylibrary)
set(LIBRARY_TYPE SHARED)

# 指定生成的共享库输出路径
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# 指定生成的共享库文件名及类型
add_library(${LIBRARY_NAME} ${LIBRARY_TYPE} ${SOURCES})

# 指定生成的共享库的安装路径
install(TARGETS ${LIBRARY_NAME} LIBRARY DESTINATION /usr/local/lib)

# 指定mylibrary.h文件的安装路径
install(FILES mylibrary.h DESTINATION /usr/local/include)
```

将 CMakeLists.txt 和头文件、源文件放在同一个文件夹 mylibrary 下。然后编译：

```bash
# 切换到项目工作目录
cd mylibrary
mkdir build && cd build
cmake .. && sudo make install
```

**方法二：使用 g++ 直接编译**

```bash
# 切换到项目工作目录
cd mylibrary
mkdir build && cd build
g++ -shared -fPIC ../mylibrary.cpp -o libmylibrary.so
```

此时编译出 `libmylibrary.so` 文件。

#### 2.3.安装

方法一中，`sudo make install` 就是安装到系统中。

方法二中，安装就是手动复制头文件和库文件到系统中。

```bash
# 切换到项目工作目录
cd mylibrary/build
sudo cp libmylibrary.so /usr/local/lib
sudo cp mylibrary.h /usr/local/include
```

#### 2.4.配置环境变量

Linux 系统通过环境变量 `$LD_LIBRARY_PATH` 寻找 `.so` 文件，可以将 `/usr/local/lib` 添加到这个环境变量中。

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
```

当 `#include "mylibrary.h"` 的时候，系统会自动从 `/usr/include`、`/usr/local/include` 等系统目录中寻找头文件。当编译的时候，系统会自动从 `$LD_LIBRARY_PATH` 目录中寻找 `.so` 文件。因此，可以手动配置环境变量。

### 3.在新项目中使用自定义库

这里我们实现一个新项目：my_demo_proj，在这个项目里，使用刚才安装的 mylibrary 库。

#### 3.1.源码编写：demo.cpp

```cpp
// #include "../mylibrary/mylibrary.h"
#include "mylibrary.h"

int main() {
    myLibraryFunction();
    return 0;
}

```

#### 3.2.编译

方法一：使用 cmake

* cmake 文件：CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.5)

project(MyProject)

# 设置C++标准为C++11
set(CMAKE_CXX_STANDARD 11)

# 添加头文件搜索路径
include_directories(/usr/local/include)

# 添加库文件搜索路径
link_directories(/usr/local/lib)

# 编译可执行文件
add_executable(demo demo.cpp)

# 链接库文件
target_link_libraries(demo mylibrary)
```

* 编译

```bash
# 切换到项目工作目录
cd my_demo_proj
mkdir build && cd build
cmake .. && make -j8
```

方法二：使用 g++

```bash
# 切换到项目工作目录
cd my_demo_proj
mkdir build && cd build
g++ ../demo.cpp -lmylibrary -o demo
```

> 编译的时候推荐使用方法一： cmake 的方法。

## 在 Linux 的特定路经下离线安装软件

由于公司的服务器不能连接外网，要使用服务器的 gpu，必须使用 docker。bevfusion 推荐的OpenMPI 版本为 4.0.4，但是使用 mpirun --version 命令可以看到，docker 容器系统里的 version 是 2.几。于是需要自己装 OpenMPI。因此要在 docker 容器中，使用离线的方式来安装。

安装的方式为

```bash
cd /opt/
# 下载 openmpi 压缩包, 
wget https://www.open-mpi.org/software/ompi/v3.0/downloads/openmpi-3.0.1.tar.gz

# 解压
tar -xvf openmpi-3.0.1.tar.gz

# 配置
./configure --prefix=/usr/local/openmpi
# 编译
make
# 安装
# MPI 库文件默认安装目录 - /usr/local/lib
sudo make install


# 环境变量设置
sudo gedit /etc/profile
# 在末尾添加下面两行行
export PATH=/usr/local/openmpi/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/openmpi/lib:$LD_LIBRARY_PATH

# 测试是否安装完成
mpirun
# 输出如下:
#--------------------------------------------------------------------------
#mpirun could not find anything to do.
#
#It is possible that you forgot to specify how many processes to run
#via the "-np" argument.
#--------------------------------------------------------------------------


# 卸载
sudo make uninstall
```

## 日期

2023/06/14：创作日期
