#PPOCR 

## 文字识别
* CTC Loss解决标签与预测不对齐的问题
* 策略
    * light backbone
      * MobileNet
    * data augementation
      * TIA类似于样条变换，获取K对个关键点，做仿射变换
    * cosine learning rate decay 
    * feature map resolution
      * 为了更多地保留垂直方向上的信息，对下采样的stride做出调整，使得垂直方向的尺寸变成原始尺寸的1/4
    * regularization parameters
    * learning rate warm-up 
    * light head
      * FPN
    * pre-train mdoel
    * PACT quantization
* 数据
    * 分类：60W
    * 文字检测：9.7W
    * 文字识别：1790W
## 文字检测

