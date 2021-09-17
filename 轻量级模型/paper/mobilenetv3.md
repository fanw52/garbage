# MobileNetV3

* 继承了mobilenetv1中的深度可分离卷积，以及moilbetv2中的逆残差结构（inverted residual network）,同时引入SE模块（通道自注意力机制），网络结构本身是通过NAS的方法得到。
* 使用h-swish替代swish，降低计算量
* $hswish=x*relu6(x+3)/6$​;  relu6，0~6之内有值，其他位置值为0
* $swish=x*sigmoid(x)$

