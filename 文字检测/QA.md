# 文字检测QA
* 部分目标重叠，推理阶段，存在重叠的两重叠目标中的其中一个置信度相对较低的现象，该如何解决？
    * 上述现象与yolov5中的mixup相似，可尝试使用mixup方法，重新train模型。

* 基于语义分割方法的文字检测模型度量方法有哪些？
  * PA:PA(pixel accuracy),与分类问题一样，计算平均准确率
  * mIoU：最小度量单位依旧是pixel，IoU_i = tp_i/(tp_i+fp_i+fn_i)
  