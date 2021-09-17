# YOLOX

### trick

* ✨simOTA（label assignment）
* ✨Anchor-free
  * Anchor依赖于数据集，泛化性不强
  * 会增加detection heads的复杂度
* ✨Decouple-head
  * yolo的解耦头表达能力有欠缺，所以采用了decoupled-head，但这种方式会带来额外的计算，因此会先用1x1的卷积对数据先降维，然后再连接两个3x3的卷积。
* ✨Augmentation
  * mixup
  * Moscia
  * 重要的是，在结束前15个epoch处停止上述两种数据增强，因为这种数据增强已经偏离了真实的自然图片分布



### 概念

* AP