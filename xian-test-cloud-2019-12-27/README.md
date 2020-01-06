# pre work
apex is need, and install it as follows:
```shell script
$ git clone https://github.com/NVIDIA/apex
$ cd apex
$ pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
$ cd ..
$ rm apex
```

jupyter
```shell script
conda install nb_conda --yes
conda install -c conda-forge jupyter_contrib_nbextensions --yes

```

[RAdam、LookAhead 双剑合璧，打造最强优化器](https://blog.csdn.net/red_stone1/article/details/101304235)

[lookahead_pytorch](https://github.com/lonePatient/lookahead_pytorch)

[radam](https://github.com/LiyuanLucasLiu/RAdam)
# history

2019-11-23 start paper

2019-10-9 add U-Net

2019-9-22 amp distributed

2019-9-19 auto-reset-lr

