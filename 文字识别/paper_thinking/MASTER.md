# MASTER(Multi-Aspect Non-local Network for Scene Text Recognition)
* 主要创新
  * 针对attention-drift以及模型并行化的问题，引入了transformer模块
  * 引入了global context模块,类似于non-local block模块，仿照transformer的multi-head形成了multi-aspect gc模块
  * 问题：中文字符识别需要采用特殊手段降低one-hot编码带来的显存占用问题。
  
    