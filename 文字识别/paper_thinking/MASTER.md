# MASTER(Multi-Aspect Non-local Network for Scene Text Recognition)

* Insight
  * 基于RNN的encode-decode文字识别结构，由于编码特征高度的相似性，导致注意力偏移的现象，
  * RNN-based model并行性（训练）较差
  * 采用transformer的解码结构，提高训练的并行性，同时由于transformer更强的编码能力，进一步缓解attention drift问题

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
      * 为了降低计算量，本篇文章使用了memery-cache的手段，存储$K_t=x_tW_K$​​与$V_t=x_tW_v$​​​ ，由于解码阶段依旧是step-by-step的计算方式，每个时刻需要计算$(XW_k)(XW_q)^T(XW_v)$​，因此，使用memery-cache手段，存储每个时刻计算的结果，可加快推理速度。
      * t=1,KQV = [$x_{t0}W_K$] [$y_{t0}Wq$​] [$x_{t0}W_V$]
      * t=1,KQV = [$x_{t0}W_K;x_{t1}W_K$​] [$y_{t1}Wq$​] [$x_{t0}W_V;x_{t1}W_V$​​]
      * ...
      * 从上面的地推计算公式可以看出，每个时刻计算需要用到历史的计算结果，此时将历史的计算结果存储在内存中，就可以降低计算速度
    
  * 问题：中文字符识别需要采用特殊手段降低one-hot编码带来的显存占用问题。
  
    