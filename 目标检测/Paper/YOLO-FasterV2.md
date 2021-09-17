# YOLO-FasterV2

* 追求速度
* 方法
  * 模型：backbone使用shufflenetv2，减少访存，模型变得更加的轻量化
  * anchor匹配机制，参考了yolov5的匹配机制，与yolov4有一定的差异
  * 检测头解耦，