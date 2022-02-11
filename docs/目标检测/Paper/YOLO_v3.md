# YOLO-v3

## yolov3

* darknet53

  * darknet19升级至darknet53，引入更深的的网络，提高模型的拟合效果，同时，采用resnet中的shortcut结构，避免梯度消失和梯度退化的问题

* fpn

  * 采用多尺度预测【v2版本中，删除了全连接层，使用多尺度训练】

* sigmoid替代softmax取消类别互斥，原因是开源数据集中存在多标签的情况，比如 “狗” 和 “动物”

  

