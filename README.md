# 使用神经网络整理照片




# 使用方法
## 下载并安装运行库
- git clone https://github.com/diandianti/photo_analyze.git
- cd phpto
- pip install -r requirements.txt
- 使用命令来测试网络能否正常运行， python test_align.py many_face.png
- 如果想要使用GPU推理，请安装onnxruntime-gpu 来代替 onnxruntime

## 使用
### 聚类分析
输入一个路径，这个目录下面包含了你要分类的图像，然后运行如下的命令。
```bash
python process.py cluster /path/to/photos \
        --out res.txt #可选参数\
        --do_copy=True #可选参数\
        --dst /path/to/copy/photos #可选参数，如果制定了do_copy那么这个必须指定\
        --log_level=1 #可选，默认为Error
```
### 比对分析
输入两个路径，第一个是源路径，即分类那些图像，第二个是底库路径，即已经分类标注好的图像路径
底库图像分布例子：
```
$ tree base
base
├── Junichiro
│   ├── Junichiro_Koizumi_0001.jpg
│   └── Junichiro_Koizumi_0012.jpg
├── Igor
│   ├── Igor_Ivanov_0001.jpg
│   └── Igor_Ivanov_0011.jpg
├── Ariel
│   ├── Ariel_Sharon_0001.jpg
│   └── Ariel_Sharon_0002.jpg
└── Sophia
    ├── Sophia_Loren_0001.jpg
    └── Sophia_Loren_0002.jpg
```

命令如下
```bash
python process.py compare /path/to/photo \
        /path/to/base \
        --dst /path/to/copy #可选参数，选择将图像复制到哪里，默认为底库路径 \
        --log_level=1 #可选，默认为Error
```


# 更改参数
默认参数可能不能满足所有人的需求，所以可以通过更改utils\config.py文件中的参数来满足整理的需求。

```python
fd = {
    "v_thd": 0.7, # 所有的人脸都会有一个置信度，即这个人脸是真实的可能性，范围在0-1， 这个值越大识别到的人脸越少，这个值越小，识别的人脸越多（可能会混进不是人脸的东西）
    "confidence_threshold" : 0.02, # 和上面差不多，不建议更改
    "top_k" : 5000, # fd后处理过程中保留置信度高的前top_k个数值
    "keep_top_k" : 750, #
    "nms_threshold": 0.4 #nms的阈值
}


cluster = {
    "dis_thd": 0.6 # 人脸之间距离的阈值，这个值越小，聚类结果的类别就越多，这个值越大，结果类别就越少
}
```

# 如果整理效果不好怎么办
- 尝试更改参数
- Fd与Fr网络我都是挑选的简单快速的网络，所以精度会差一点，后续会添加精度更高的网络，但是运行速度就不知道会如何了

# 网络模型
- FD：retinaface (https://github.com/biubug6/Pytorch_Retinaface)
- Fr: Facenet (https://github.com/yywbxgl/face_detection/tree/master/models)

# 功能 & Todo
- [x] 聚类分析
- [x] 比对分析
- [ ] 多类别分析
- [ ] 年龄分析
- [ ] 图像摘要
- [ ] GUI
