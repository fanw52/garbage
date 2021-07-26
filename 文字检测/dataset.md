[ICDAR2015](https://rrc.cvc.uab.es/?ch=4)
* 适用任务：英文检测识别
* 数据情况：英文，训练1000，测试500
* 标注形式：（x1,y1,x2,y2,x3,y3,x4,y4,transcription）
* 链接：https://pan.baidu.com/s/1MovnmtUmEtQFFOvbFdpgDQ  密码: g59m

[MLT2019](https://rrc.cvc.uab.es/?ch=15)
* 适用任务：多语言检测识别（美团）
* 数据情况：英文，训练10000，测试10000
* 标注形式：（x1,y1,x2,y2,x3,y3,x4,y4,transcription）
* 链接: https://pan.baidu.com/s/1vQ3wTrhERwa-o9XmmIUDEg  密码: 6d0j

[COCO-Text_v2](https://bgshih.github.io/cocotext/)
* 适用任务：多语言检测识别
* 数据情况：英文，训练43686，验证10000，测试10000
* 标注形式：
* 链接: https://pan.baidu.com/s/1ab2QQfM564thC3MBO56AIw  密码: 06qo


[ReCTS](https://rrc.cvc.uab.es/?ch=12&com=introduction)
* 适用任务：多语言检测识别
* 数据情况：英文，训练20000，测试5000
* 标注形式：
    ```json
    {
        “chars”: [
            {“points”: [x1,y1,x2,y2,x3,y3,x4,y4], “transcription” : “trans1”, "ignore":0 },
            {“points”: [x1,y1,x2,y2,x3,y3,x4,y4], “transcription” : “trans2”, " ignore ":0 }],
        “lines”: [
            {“points”: [x1,y1,x2,y2,x3,y3,x4,y4] , “transcription” : “trans3”, "ignore ":0 }],
    }
    ```
* 链接: https://pan.baidu.com/s/1il2vy5ryvzRgolo0v8evRA  密码: rm7l


[ArT(包含Total-Text和SCUT-CTW1500)](https://rrc.cvc.uab.es/?ch=14)
* 适用任务：多语言检测识别
* 数据情况：多语言，训练5,603，测试4,563，包含扭曲文字
* 标注形式：（x1,y1,x2,y2,x3,y3,x4,y4,transcription）
  ```json
    {
    “gt_1”:  [   
       {“points”: [[x1, y1], [x2, y2], …, [xn, yn]], 
        “transcription” : “trans1”, 
        “language” : “Latin”, 
        "illegibility": false },
                    
        {“points”: [[x1, y1], [x2, y2], …, [xn, yn]], 
          “transcription” : “trans2”, 
          “language” : “Chinese”, 
          "illegibility": false }
    ],
    }
  ```
* 链接: https://pan.baidu.com/s/1oFucFuuWt15OWU4wz3EeFA  密码: hvap


[LSVT](https://rrc.cvc.uab.es/?ch=16)
* 适用任务：多语言检测识别
* 数据情况：英文，训练30000，测试20000 (45w街景图片，5w有强标签)
* 标注形式：
  ```json
    {
    “gt_1”:  [   
      {“points”: [[x1, y1], [x2, y2], …, [xn, yn]], 
      “transcription” : “trans1”, 
      "illegibility": false },
                    
      {“points”: [[x1, y1], [x2, y2], …, [xn, yn]], 
      “transcription” : “trans2”, 
       "illegibility": false }],
    }
  ```
* 链接: https://pan.baidu.com/s/1qyAipd6lt73vDtxfK628uQ  密码: dpqh


[icdar2017rctw](https://blog.csdn.net/wl1710582732/article/details/89761818)
* 适用任务：多语言检测识别
* 数据情况：英文，训练10000，测试10000(手机拍照数据+互联网数据)
* 标注形式：(x1,y1,x2,y2,x3,y3,x4,y4,<识别难易程度>,transcription)
* 链接: https://pan.baidu.com/s/11nNHKZG1yF5ZAw4NiXV9pQ  密码: c108


MTWI 2018
[检测](https://tianchi.aliyun.com/competition/entrance/231685/introduction)
[识别](https://tianchi.aliyun.com/competition/entrance/231684/introduction)
* 适用任务：多语言检测识别（淘宝电商合成图片）
* 数据情况：英文，训练10000，测试10000
* 标注形式：（x1,y1,x2,y2,x3,y3,x4,y4,transcription）
* 链接: https://pan.baidu.com/s/1CMoel0CoLR4wOOhsf-nH4g  密码: o7pr


[百度中文场景文字识别](https://aistudio.baidu.com/aistudio/competition/detail/20)
* 适用任务：多语言识别
* 数据情况：训练21w，测试8w
* 标注形式：（h,w,name,value）
* 链接: 链接: https://pan.baidu.com/s/1xdl_xhjOHeDY6M-C0KpEDQ  密码: kt0v


[CTW](https://ctwdataset.github.io/)
* 适用任务：检测识别（街景）
* 数据情况：英文，3w
