# 文字识别QA

[TOC]

#### 重要的事说三遍：数据！数据！数据！

* 身份证文字识别问题？
    * 身份证文字文字识别主要体现在身份证号码识别，以及生僻字识别
      * 身份证号码的问题
        * 身份证号码最后一位丢失，合成身份证号码图片数据，融合历史数据，对模型进一步微调，解决身份证号码识别问题
      * 生僻字识别问题
        * 生僻字识别问题是身份证识别的一个常见的问题，为了支持生僻字识别，我们生成了包含部分生僻字的数据，融合历史数据训练文字识别模型，
          从结果开看，基于生成数据训练的模型在真实场景下也有不错的表现。
* 如何理解s2s方法中attetnion drift问题？
  * 现象定义：feature与target不能对齐(或者是错误对齐)，如：19-G&19-1&19-2预测为19-G&19-1&19-1&19-2或19-G&19-2
    * 19-G&19-1&19-2--> 19-G&19-2: 预测第二个19时，认为是输入特征的后半部分，导致中间部分没有解码出来
    
  * 具体可以通过下图来理解
    ![attention drift](./data/attention%20drift.png)
    * 字符"M"的注意力权重集中在"K"上导致解码结果丢失"M"
    * 总的来说，注意力权重的错误分配，使得原始特征的显著性降低，进而使得解码结果出现错误
    * 编码特征高度相似，导致在RNN-based local attention下出现attention confusion的现象（MASTER给出的解释），结论就是用编码能力更强的网络去替代lstm
* 文字识别的一些优化方案：
* label smoothing （ctc算法的尖峰分布问题？）
* 生层与应用场景相关性更高的数据
* CTC解码过程中出现多字漏字的现象如何解决？
  * 可能字符不均匀
* 弯曲文字纠正比较好的算法？
  * ABCNet
* 基于注意力机制的损失函数：
    * 每个time step交叉熵的加权平均
* attention解码
    * decoder中，每个时刻的解码状态和encoder中的隐藏状态计算attention，并对encoder隐藏层加权求和，得到decoder其中一个输入，另外一个输入是前一个时刻的解码结果，见下图：	![image-20220109175639846](QA.assets/image-20220109175639846.png)

#### CTC

* 简单描述CTC Loss?
  * 求出所有可能表示为目标序列的组合情况，最大化序列和【负对数】
* 
