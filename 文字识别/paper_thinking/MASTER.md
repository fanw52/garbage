# MASTER(Multi-Aspect Non-local Network for Scene Text Recognition)

* 模型结构
  ![模型结构](../data/MASTER.png)

* 主要创新
  * 针对attention-drift以及模型并行化的问题，引入了transformer模块
  * 引入了global context模块,类似于non-local block模块，仿照transformer的multi-head形成了multi-aspect gc模块
    * Non-Local Block
      ![示意图](../data/non-local-block.png)
      
    * multi-aspect gc block
      ![示意图](../data/multi-aspect-gcAttention.png)
      
    * gc block是non-local block的简化版，前者用一个1x1的卷积沿着channel将feature map降到一维，计算注意力权重，
      后者利用两个1x1的卷积计算注意力权重。
      
  * Decoder
    * transformer的解码与lstm的解码类似，用encoder部分的输出计算注意力权重，并对输入向量重新分配，获得当前时刻的输入
    * 从模型结构示意图可以看出，在解码阶段
      * 首先用multi-head attention计算当前时刻的输入编码，得到tmp_feature,相当于LSTM中将时刻t-1的输出输入到模型中
      * 随后获取encoder部分的输出，分别作为Key和quary,计算获得注意力权重，并利用该权重对tmp_feature（value）重新分配
    
  * 问题：中文字符识别需要采用特殊手段降低one-hot编码带来的显存占用问题。
  
    