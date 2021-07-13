# RARE(Robust scene recognition with Automatic REctification)

* 目的：解决不规则文本的识别问题
* 模型
    * STN(Spatial Transformer Network)
      * Localization Network
        * 回归K对样条变换点
          
      * Grid Generator
        * 估计TPS变换矩阵（用于计算新的坐标位置，新的坐标可能是非整数点，可由周围几个）
          
      * Sampler
        * 点p'【变换后的点】由点p【变换前的点】线性插值得到
        
    * SRN(Sequence Recognition Network)
        * attention-based model
        