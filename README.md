# 使用神经网络整理照片

一个可以用来整理照片的工具（我也是提交了patch之后才发现我把photo打错了，就这样吧）


# 使用方法
- git clone https://github.com/diandianti/phpto.git
- cd phpto
- pip install -r requirements.txt
- 使用命令来测试网络能否正常运行， python test_align.py many_face.png
- 处理图像： python process.py cluster /path/to/photos
- 如果想要使用GPU推理，请安装onnxruntime-gpu 来代替 onnxruntime


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
- [ ] 比对分析
- [ ] 多类别分析
- [ ] 年龄分析
- [ ] 图像摘要
- [ ] GUI
