# PaddleSpeech

* PaddleSpeech集语音识别、语音合成、语音翻译、语音分类等能力为一体的开源项目

## 快速开始

* 假如只想白嫖已经训练好的模型，那么就按照下面的步骤走吧！

#### 构建环境

* 拉镜像： `nvidia-docker pull registry.baidubce.com/paddlepaddle/paddle:2.2.0-gpu-cuda10.2-cudnn7`

* 获取源码：`git clone https://github.com/PaddlePaddle/PaddleSpeech.git`

* 进入容器内部创建环境： `nvidia-docker run --net=host --ipc=host --rm -it -v $(pwd)/PaddleSpeech:/PaddleSpeech registry.baidubce.com/paddlepaddle/paddle:2.2.0-gpu-cuda10.2-cudnn7 /bin/bash`

* 找到源码: `cd /PaddleSpeech`

* 安装pytest-runner: `pip install pytest-runner -i https://pypi.tuna.tsinghua.edu.cn/simple `

* 安装依赖： `pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`

* 综上Dockerfile如下：

* ```dockerfile
  FROM registry.baidubce.com/paddlepaddle/paddle:2.2.0-gpu-cuda10.2-cudnn7
  WORKDIR /PaddleSpeech
  RUN wget https://paddlespeech.bj.bcebos.com/Parakeet/tools/nltk_data.tar.gz && tar -zxvf nltk_data.tar.gz && mv nltk_data ~ && rm -f nltk_data.tar.gz
  COPY . .
  RUN apt-get install libsndfile1 -y
  RUN pip install pytest-runner -i https://pypi.tuna.tsinghua.edu.cn/simple
  RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```

* 这里遇到的大坑就是执行infer.py脚本，会检测是否已经下载了nltk_data，如果没有下载，就会在线下载，这里往往非常耗时，因此需要事先下载好放到~下，一般是家目录

* 参考链接：https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/install_cn.md

###  模型下载

**ASR** 

 `https://paddlespeech.bj.bcebos.com/s2t/wenetspeech/asr1_conformer_wenetspeech_ckpt_0.1.1.model.tar.gz`

其中：

````
ckpt_path = "model/asr1_conformer_wenetspeech_ckpt_0.1.1/exp/conformer/checkpoints/wenetspeech"
config="./model/asr1_conformer_wenetspeech_ckpt_0.1.1/model.yaml"
````

**TTS**

 `https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_nosil_baker_ckpt_0.4.zip`

`https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwg_baker_ckpt_0.4.zip` 

其中: 

```text
am="fastspeech2_csmsc"
am_config="model/fastspeech2_nosil_baker_ckpt_0.4/default.yaml"
am_ckpt="model/fastspeech2_nosil_baker_ckpt_0.4/snapshot_iter_76000.pdz"
am_stat="model/fastspeech2_nosil_baker_ckpt_0.4/speech_stats.npy"
phones_dict="model/fastspeech2_nosil_baker_ckpt_0.4/phone_id_map.txt"
voc="pwgan_csmsc"
voc_config="model/pwg_baker_ckpt_0.4/pwg_default.yaml"
voc_ckpt="model/pwg_baker_ckpt_0.4/pwg_snapshot_iter_400000.pdz"
voc_stat="model/pwg_baker_ckpt_0.4/pwg_stats.npy"
```



### 启动docker容器

启动命令：

`nvidia-docker run --net=host --ipc=host --rm -it -v /data/wufan/PaddleSpeech:/PaddleSpeech -v /model:/model paddle_speech:v1.4.0 /bin/bash`