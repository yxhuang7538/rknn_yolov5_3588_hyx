# RK3588上部署YOLOv5-DeepSORT红外目标跟踪模型

## 1、结构
- include
    - global.h
    声明了所需的全局变量，以及需要使用的全局函数。

- model
    - labels.txt 存放目标类别
    - xxx.mp4 需要检测的视频
    - xxx.rknn 需要用到的rknn模型

- src 
    - detection.cc 实现目标检测
    - global.cc 定义全局变量
    - main.cc 主函数
    - postprocess.cc 实现检测后处理
    - trackprocess.cc 实现目标跟踪
    - video_io.cc 实现视频读取和存储

## 2、快速应用
### 2.1 前期准备
首先需要用RKNN-Toolkit2工具将训练模型转为RKNN模型。

得到转换模型后，可以选择rknpu2提供的接口在RK平台进行开发应用。

需要准备的库文件
- RKNN API : rknpu2/runtime/librknnrt.so

```
git clone https://github.com/rockchip-linux/rknpu2.git
cd rknpu2/examples
git clone https://github.com/yxhuang7538/rknn_yolov5_3588_hyx.git
cd rknn_yolov5_3588_hyx
# 修改CMakeLists.txt中你的opencv的路径
# 修改build-linux_RK3588.sh中编译器路径
./build-linux_RK3588.sh
cd install/rknn_yolov5_3588_hyx_linux
./rknn_yolov5_3588_hyx v ./model/xxx.rknn ./model/xxx.mp4
```

## 3、进度
- [ ] 采用零拷贝API接口框架
- [ ] 采用通用API接口框架
- [x] 实现目标检测
- [x] 实现目标跟踪
- [ ] 优化目标检测
- [ ] 优化目标跟踪
- [ ] 修复Resize问题
- [ ] 提高帧率
    - [ ] 使用蒸馏模型
    - [x] 去掉Focus层
    - [x] int8量化
    - [ ] 多线程权重复用

## 4、参考
1. https://github.com/rockchip-linux/rknpu2
