# TensorRT

* [TensorRT、C++推理](https://github.com/wang-xinyu/tensorrtx)
* 以[yolov5 v6.0](https://github.com/ultralytics/yolov5/tree/v6.0)为例，进行工程实现

* Docker镜像地址：https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt/tags


* 使用步骤
  * 使用模型文件创建engine，序列化后存储到本地
  * 使用C++ parser创建tensorRT网络
  * 用c++