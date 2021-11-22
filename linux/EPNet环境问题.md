### EPNet碰到的问题

环境：

nvidia-smi：11.2；

cuda：11.5；

 pytorch：1.2



编译pointnet++等库*



###### 问题1：Provided PTX was compiled with an unsupported toolchain. 

解决：查了半天资料，说是cuda版本太超前。换11.2，11.1都不行。最终换成nvcc：10.1；cudatoolkit:10.1



###### 问题2：ImportError: libcudart.so.11.1: cannot open shared object file: No such file or directory

疑惑：想不明白，为啥我都换成10.1了，还报个11.1的错？

原因：因为我在*这一步，已经用cuda11.1的环境编译（setup.py）了，所以报11.1的错

解决：把各种和setup.py同级目录下的build/ dist/ 这种文件夹删干净，然后重新编译(setup.py)

