
# RSSRAI
遥感土地图像分割
![任务展示](img/图例.jpg)
## 离线数据增强
原图共十张，每张图切割16张，共得到160张原图
### 训练集增强
抽取145张原图
对label像素统计，像素十分不均衡，通过离线调整原图数量，使少样本类别分布尽量相同
![原始分布](img/原始分布.png)
![数据增强](img/增强分布.png)
1水田,2水浇地,3旱耕地,4园地,5乔木林地,6灌木林地,7天然草地,8人工草地,9工业用地,10城市住宅,11村镇住宅,12交通运输,13河流,14湖泊,15坑塘

一共16类，背景类未统计

### 验证集划分
抽取15张原图
进一步切分
## apex
```angular2
$ git clone https://github.com/NVIDIA/apex
$ cd apex
$ pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./