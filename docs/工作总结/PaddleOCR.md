# PaddleOCR使用

[TOC]

## 文字识别

### 训练

* 配置yaml文件：rec_mv3_none_bilstm_ctc.yaml

  * 主要修改下面几行代码

  * ```python
    name: SimpleDataSet
    data_dir: /data/wufan/data/database/image/baidu
    label_file_list: ["/data/wufan/data/database/image/baidu/train.txt"]
    ```

* 图片数据数据

  * ```
    .
    ├── baidu
    │   ├── train_images
    │   │   ├── img_0.jpg
    │   │   ├── img_100000.jpg
    │   │   ├── img_100001.jpg
    │   │   ├── img_100002.jpg
    │   │   ├── img_100003.jpg
    │   │   ├── img_100004.jpg
    │   │   ├── img_100005.jpg
    ```

* 标签数据

* ```
  ./train_images/img_0.jpg	福
  ./train_images/img_1.jpg	空调维修保养
  ./train_images/img_2.jpg	空调移机保养
  ./train_images/img_3.jpg	久斯台球会所
  ./train_images/img_4.jpg	熟面条混
  ./train_images/img_5.jpg	ＡＮＵＦＡＣＴＵＲＩＮＧＣＯ．，ＬＴ
  ./train_images/img_6.jpg	上海创科泵业制造有限公司
  ./train_images/img_7.jpg	鲜面条饺
  ./train_images/img_8.jpg	木炭
  ./train_images/img_9.jpg	雅迪
  ./train_images/img_10.jpg	三黄鸡柴鸡鲜鱼店
  ```

* 训练指令

* ```
  python3 -m paddle.distributed.launch --gpus '0,1,2,3'  tools/train.py -c configs/rec/rec_mv3_none_bilstm_ctc.yaml
  ```

* 

## 文字检测

