# 基础知识

* 残差连接的作用？
    * 防止梯度消失【反向传播】和【特征退化】
  
* 卷积
  * 定义
    * 
  * 性质
    * 是一种局部特征提取器
    
  * 卷积的变体
    * group conv
      * 降低卷积计算的浮点运算量
    * deformable conv
      * 传统发的卷积计算方法拥有固定的几何结构，对几何形变的物体建模受限，而dfc通过学习额外的偏移量，使得卷积核在
        input feature map的采样点上发生偏移，实现在当前位置周围随意采样，由于位置的随意性，会出现非整数点的情况，
        而这种情况的值，需要通过双线性插值的方法实现。
    
  * 感受野
    
    
  * 输出尺寸
    * ceil(w-k+2p)/s+1
    
  * 卷积实际计算的方式
  

* 循环神经网络
  * 定义
    * 
  * 性质
    * 
  * 变体
    * lstm
      * 性质
      * 相关：transformer
  
* 注意力机制
  * 定义
    * 通过某种方式计算获取输入特征之间的关联关系，这种关系的强弱就是注意力机制
  * 性质
    * 
  
* MobileNet
  * 深度可分离卷积(depthwise separable conv)
    * 卷积核与input feature map对应通道做卷积操作
    * 逐点卷积（pointwise convolution：1x1）
  * 深度可分离卷积+逐点卷积（计算量：K\*K\*W\*H\*Ci+K\*K\*W\*H）替代普通卷积(计算量K\*K\*W\*H\*Ci\*Co),计算量约为原先的1/8
  
  
*  MobileNetV2
  * inverted residuals（升维，卷积，降维），保留更多信息
  * depthwise separable conv
  * pointwise conv
  
  * common residual(降维，卷积，升维)

* MobileNetV3
  * depthwise separable conv
  * inverted residuals
  * SE模块（squeeze and excitation）
    * squeeze
      * 对W\*H\*C的feature map做global avg pooling得到1x1xC的特征图
    * excitation
      * 对1x1xC的feature map做非线性变换，并结合sigmoid得到每个通道的注意力权重（channel-level），增强重要特征，抑制不重要特征
  * h-swish(x\*sigmoid(beta\*x)),平滑，非单调，无上界，有下界