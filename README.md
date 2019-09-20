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
# history

2019-9-19 auto-reset-lr