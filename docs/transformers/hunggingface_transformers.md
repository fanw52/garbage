# hunggingface_transformers

[TOC]

## 注意

* 文字识别
  * TrOCR
* 图像分类
  * Vision Transformer(ViT) 
  * data-efficient image transformers(DeiT)
* 目标检测
  * DETR （detection with transformer）
* 版面分析
  * LayoytLM
  * LayoutLM_v2
  * LayoutXLM
* NLP
  * ALBERT
  * RoBERTa
  * BERT

## 代码理解

* Huggingface/transformers核心为三个部分
  * config:AutoConfig
    * 初始化模型参数
  * token: AutoFeatureExtractor; AutoTokenizer
    * 编码输入序列
  * model:AutoModelForImageClassification;AutoModelForTokenClassification
    * 初始化模型

* 本地数据加载[json格式]

  * 文本分类任务：

    * ```json
      {"label": "like", "text": "又要重复我每年必说的台词了，2010就要过去了，我很怀念它。"}
      {"label": "like", "text": "至少，它看上去很美。"}
      ```

    * 调用形式：

      * ```python
        raw_datasets = load_dataset("json", data_files=data_files, cache_dir=model_args.cache_dir)
        def preprocess_function(examples):
          # Tokenize the texts
          args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
          )
          result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)
        
          # Map labels to IDs (not necessary for GLUE tasks)
          if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
           return result
        
        ```

  * 序列标注任务

    * ```json
      {"tokens": ["生", "生", "不", "息", "C", "S", "O", "L", "生", "化", "狂", "潮", "让", "你", "填", "弹", "狂", "扫"], "tags": ["O", "O", "O", "O", "B-g
      ame", "I-game", "I-game", "I-game", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]}
      {"tokens": ["那", "不", "勒", "斯", "v", "s", "锡", "耶", "纳", "以", "及", "桑", "普", "v", "s", "热", "那", "亚", "之", "上", "呢", "？"], "tags": [
      "B-organization", "I-organization", "I-organization", "I-organization", "O", "O", "B-organization", "I-organization", "I-organization", "O", "O", "B-o
      rganization", "I-organization", "O", "O", "B-organization", "I-organization", "I-organization", "O", "O", "O", "O"]}
      ```

    * 调用形式

      * ```python
        raw_datasets = load_dataset("json", data_files=data_files, cache_dir=model_args.cache_dir)
        ```

  * 图像分类

    * ```json
      {"image": {"bytes": null, "path": "/data/wufan/algos/data/transformers_data/cat_vs_dog/train/cat.8197.jpg"}, "labels": "cat"}
      {"image": {"bytes": null, "path": "/data/wufan/algos/data/transformers_data/cat_vs_dog/train/dog.2484.jpg"}, "labels": "dog"}
      ```

  * 调用形式

    * ```python
      ds = load_dataset("json", data_files=data_files, cache_dir=model_args.cache_dir)
      def train_transforms(example_batch):
        	"""Apply _train_transforms across a batch."""
        	example_batch["pixel_values"] = [
          _train_transforms(Image.open(example["path"]).convert("RGB")) for example in example_batch["image"]
        ]
        example_batch["labels"] = [int(label2id[l]) for l in example_batch["labels"]]
        return example_batch
      ```

    * 

  * 训练指令

    * 文本分类

      * ```bash
        python3 ./examples/pytorch/text-classification/run_glue.py\
          --model_name_or_path /data/model/bert-base-uncased \
          --train_file /data/nlpcc2014/nlpcc2014Train.json \
          --validation /data/nlpcc2014/nlpcc2014Val.json \
          --test_file /data/nlpcc2014/nlpcc2014Test.json \
          --do_train  \
          --do_eval  \
          --do_predict  \
          --max_seq_length 128 \
          --per_device_train_batch_size 32 \
          --learning_rate 2e-5  \
          --num_train_epochs 1 \
          --output_dir /data/nlpcc2014/workdir/ \
          --overwrite_output_dir
        ```

    * 序列标注

      * ```bash
        python3 ./examples/pytorch/token-classification/run_ner.py \
          --model_name_or_path /data/model/albert-tiny-chinese \
          --train_file /data/cluener/CLUENER2020_train.json \
          --validation_file /data/cluener/CLUENER2020_dev.json \
          --test_file /data/cluener/CLUENER2020_test.json \
          --output_dir /data/cluener/workdir/albert-tiny-chinese/ \
          --do_train  \
          --do_eval \
          --do_predict  \
          --num_train_epochs 1 \
          --overwrite_output_dir
        ```

  * 图像分类

    * ```json
      docker run -it --gpus='"device=2,3"' -v /data/wufan/algos/data/transformers_data:/data/wufan/algos/data/transformers_data transformers:v1.0.4 bash -c "python3 examples/pytorch/image-classification/run_image_classification.py --output_dir output_dir --model_name_or_path /data/wufan/algos/data/transformers_data/model/tiny-random-vit --train_file /data/wufan/algos/data/transformers_data/cat_vs_dog_train.json --do_train --do_eval --learning_rate 1e-4 --per_device_train_batch_size 32 --per_device_eval_batch_size 2 --remove_unused_columns False --overwrite_output_dir True --dataloader_num_workers 64 --metric_for_best_model accuracy --max_steps 10000 --train_val_split 0.1 --seed 42"
      
      ```

    * Albert / bert-base-uncased

## 多模态

* layoutLM

* 为什么不选择这个算法

  * 从模型的角度，该算法非常重，实际使用起来比较困难
  * 从数据的角度，标注复杂，需要标注单字位置信息，实体信息等。

* 输入，单字坐标，字符位置，word embedding

* 预训练任务

  * image-token-mask: 预测被mask掉的字
  * Text-image alignment：预测文本（单词）是否被覆盖
  * Text-image matching：预测图片和文本是否匹配

* 工程修改

  * 执行指令

    ```python
    python examples/run_xfun_ser.py--model_name_or_path /data/wufan/algos/data/transformers_data/model/layoutxlm-base --dataset_name /data/wufan/algos/data/transformers_data/XFUN --output_dir /tmp/test-ser --do_train --do_eval --lang zh --max_steps 1000  --fp16 --per_device_train_batch_size 4 --preprocessing_num_workers 32 --overwrite_output_dir
    ```

  * --model_name_or_path: 本地存储的预训练模型路径

  * --dataset_name: 本地存储的数据路径

* 部分修改

  * 考虑到开源的项目通过加载互联网数据进行模型训练，而我们实际使用时，需要加载本地离线数据，因此，需要对工程做一些修改，使得可以加载离线数据

  * 修改

  * 加载本地评价方法

  * ```python
    metric = load_metric("./metric/seqeval.py")
    ```

  * 额外传入图像数据根目录

  * ```python
    datasets = load_dataset(
        os.path.abspath(layoutlmft.data.datasets.xfun.__file__),
        f"xfun.{data_args.lang}",
        data_dir =  data_args.dataset_name, # 额外传入
        additional_langs=data_args.additional_langs,
        keep_in_memory=True,
    )
    ```

  * 重构函数_split_generators

  * ```python
        def _split_generators(self, dl_manager):
            """Returns SplitGenerators."""
            train_files_for_many_langs = [{"label":os.path.join(f"{self.config.data_dir}", f"{self.config.lang}.train.json"),
                                          "img_dir":os.path.join(f"{self.config.data_dir}", f"{self.config.lang}.train")}]
            val_files_for_many_langs = [{"label":os.path.join(f"{self.config.data_dir}", f"{self.config.lang}.val.json"),
                                        "img_dir":os.path.join(f"{self.config.data_dir}", f"{self.config.lang}.val")}]
            return [
                datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepaths": train_files_for_many_langs}),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION, gen_kwargs={"filepaths": val_files_for_many_langs}
                ),
                # datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepaths": test_files_for_many_langs}),
            ]
    
    ```

    * 加载本地数据位置，提供给函数_generate_examples使用，解析的时候通过key-value确定数据