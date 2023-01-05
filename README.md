# 【飞桨学习赛：百度搜索首届技术创新挑战赛：赛道一】第5名方案
> 【飞桨学习赛：百度搜索首届技术创新挑战赛：赛道一】第5名方案(任务一)
>  方案思路请结合---百度挑战赛答辩.pptx--一起理解
```
子任务 1：答案抽取 具体描述请查看竞赛链接
https://aistudio.baidu.com/aistudio/competition/detail/660/0/task-definition
```
## 1. 项目描述
### 1.1 项目任务简要介绍：
```
给定query，以及doc_text，预测出query对应的答案，可能是多个，或者没有答案
example:
    问题q：备孕偶尔喝冰的可以吗
    篇章d：备孕能吃冷的食物吗 炎热的夏天让很多人都觉得闷热...,下面一起来看看吧! 备孕能吃冷的食物吗 在中医养生中,女性体质属阴,不可以贪凉。吃了过多寒凉、生冷的食物后,会消耗阳气,导致寒邪内生,侵害子宫。另外,宫寒是肾阳虚的表现,不会直接导致不孕。但宫寒会引起妇科疾病,所以也不可不防。因此处于备孕期的女性最好不要吃冷的食物。 备孕食谱有哪些 ...
    答案a：在中医养生中,女性体质属阴,不可以贪凉。吃了过多寒凉、生冷的食物后,会消耗阳气,导致寒邪内生,侵害子宫。另外,宫寒是肾阳虚的表现,不会直接导致不孕。但宫寒会引起妇科疾病,所以也不可不防。因此处于备孕期的女性最好不要吃冷的食物。
```
### 1.2 模型主要思路：
```
详细见: 百度挑战赛答辩.pptx
1) 对抗扰动的训练
   主要对FGM以及pgd以及AWP进行了测试，在静态图进行扰动，目前paddle没有找到相关代码，是自己按照动态图进行编写的代码;目前AWP在本此训练并
   没有很成功，一直训练loss为nan，有可能对embedding的扰动过大，PGD效果虽然有一定的提升，但是占用显存明显过大，因此最后选用的时fgm，提
   升约为0.03左右.
2) 数据的探索
   经过统计发现，len(query)+len(doc_text)超过512长度占比 0.55,答案长度超过512 竟然占0.44,doc_text长度分布比较均匀，所以一定要
   仔细处理doc的长度，而不是简单的截断.
   (1)
      首先对预测文本doc_text,按照最大文本512 进行分割，分别预测，最后答案进行合并/ernie_dqa_task1/applications/tasks/sequen
      ce_labeling/data/data_deal.py
      原始test文本3868个样本
      a. test_data_split函数
         主要是把doctext按照512分割，以id为同一个文档为唯一标识，为了后来把同一文档答案合并做唯一标志分割后9631个样本   
      b. 按照正常的预测模型进行推理得到最后的text答案 4000.txt 行数为顺序的9631个样本
      c. predict_deal函数 把结果按照id对答案进行合并最终在原始模型分数为0.61109提升为0.62846,训练步数为4000步0.628，充分说明如
         何处理超过512的长文本是至关重要的，不能简单的截断.
   (2)
      继承步骤（1）虽然这种强制截断有损失，但是还有一定的提高，如果训练数据同样的方法纪念性操作，获取预测相同的数据分布，应该有所提升，这
      一步训练数据采用相同的方式进行处理.实际效果很差，就舍弃了，因为信息截断对模型效果影响非常大
   (3)
       如果使用简单的截断训练数据，则会损失文档截断的答案标注信息，因此使用了以答案为中心的阶段策略，重新构造新的训练数据，但是效果并不明
       显，其实如果有时间是值得研究的.
3) 大模型的加持
    使用了ernie3.0_x_base用相同方法提高到0.64993，具有大模型的优势效果这也充分说明，如果有足够的数据提供新的预训练方式是提高阅读理解
    效果的好的方式，设计新的预训练方式
4) 使用命名实体识别的方式对答案进行抽取
   使用命名实体识别方法
       a.构造B-ans I-ans E-ans  O对应的标签训练数据
       并且使用start-end 加BIE预测联合损失，利用start  end预测 初步增加为0.65143
       并且使用start-end 加BIE预测联合损失，直接利用soft_max 预测命名实体标注，直接为0.67;利用联合训练即对之前预测
       start-end任务有帮助0.65143 ，更是对entity_recgnzie的结果有帮助,因此转化为实体识别任务是有很大的帮助的
       
       b.直接使用全实体识别任务进行训练，采用beam-search进行测试,
       暂时发现纯crf并不能发挥作用，直接使用crf 准确率增高了，但是召回率降低了，说明严格了，所以可以考虑  start与crf结合的方法
       目前分数是0.679
5) 分段attention
   分段attention主要是长文档依次输入模型，对于依赖具有一定的考验性，因此分为两段，第二段利用第一段的信息作为key以及value，利用第二段的
   query选择性的把第一段的value信息添加到第二段中，指导第二段的答案预测，最终单个模型预测为0.68025.
6) 最终模型融合，简单的使用三个模型，进行1：2投票，最终提交结果为0.689
```
## 2.项目结构
> 项目结构如下:
```
├── applications
│   ├── __init__.py
│   ├── models_hub  #用来存储预训练模型的路径，例如：ernie_3.0_base_ch_dir
│   └── tasks
│       ├── __init__.py
│       └── sequence_labeling
│           ├── begin_train.sh #项目启动训练脚本
│           ├── data  #验证数据集，训练数据集，测试数据集等
│           │   ├── data_deal.py #数据处理以及提交最后文件等代码
│           │   ├── data_static.py
│           │   ├── dev_data
│           │   ├── test_data
│           │   └── train_data
│           ├── dict 
│           │   ├── vocab_label_map.txt
│           │   └── vocab.txt
│           ├── evaluate.py
│           ├── evaluate.sh
│           ├── examples #模型训练以及模型推理的参数json文件
│           │   ├── seqlab_ernie_fc_ch_infer.json
│           │   └── seqlab_ernie_fc_ch.json
│           ├── final_submit.py  #生成最后提交文件的代码
│           ├── final_submit.sh  #生成最后提交文件的代码
│           ├── inference
│           │   ├── custom_inference.py
│           │   └── __init__.py
│           ├── log #训练测试日志
│           │   ├── test.log
│           │   ├── test.log.2022-11-24
│           │   ├── test.log.wf
│           │   └── test.log.wf.2022-11-24
│           ├── model #模型结构代码
│           │   ├── ernie_fc_sequence_label.py
│           │   ├── __init__.py
│           │   └── t.py
│           ├── output #模型保存地方
│           │   ├── 4000.txt
│           │   ├── 4001.txt
│           │   ├── seqlab_ernie_3.0_base_fc_ch
│           │   │   ├── save_checkpoints
│           │   │   └── save_inference_model
│           │   └── subtask1_test_pred_ner.txt
│           ├── preprocess
│           │   └── dev.txt
│           ├── preprocess.py
│           ├── run_infer.py
│           ├── run_trainer.py
│           └── trainer
│               ├── custom_dynamic_trainer.py
│               ├── custom_trainer.py
│               └── __init__.py
├── erniekit  #提供baseline的原有文件，其中对抗训练代码在static_trainer.py，数据处理代码在ernie_text_field_reader.py文件中
│   ├── build.sh
│   ├── ci.yml
│   ├── common
│   │   ├── __init__.py
│   │   ├── jit_wenxin.py
│   │   ├── register.py
│   │   └── rule.py
│   ├── controller
│   │   ├── dynamic_trainer.py
│   │   ├── evaluate.py
│   │   ├── inference.py
│   │   ├── __init__.py
│   │   ├── static_trainer_ernie_gen.py
│   │   └── static_trainer.py
│   ├── data
│   │   ├── data_set_ernie3.py
│   │   ├── data_set.py
│   │   ├── data_set_reader
│   │   │   ├── base_dataset_reader.py
│   │   │   ├── basic_dataset_reader.py
│   │   │   └── __init__.py
│   │   ├── field.py
│   │   ├── field_reader
│   │   │   ├── base_field_reader.py
│   │   │   ├── custom_text_field_reader.py
│   │   │   ├── ernie_classification_field_reader.py
│   │   │   ├── ernie_seqlabel_label_field_reader.py
│   │   │   ├── ernie_text_field_reader_for_doc.py
│   │   │   ├── ernie_text_field_reader_for_multilingual.py
│   │   │   ├── ernie_text_field_reader.py
│   │   │   ├── __init__.py
│   │   │   ├── scalar_array_field_reader.py
│   │   │   ├── scalar_field_reader.py
│   │   │   └── text_field_reader.py
│   │   ├── __init__.py
│   │   ├── reader_config.py
│   │   ├── tokenizer
│   │   │   ├── custom_tokenizer.py
│   │   │   ├── doie_basic_tokenizer.py
│   │   │   ├── doie_ernie_tiny_tokenizer.py
│   │   │   ├── ernie_sim_slim_tokenizer.py
│   │   │   ├── __init__.py
│   │   │   ├── lac_tokenizer.py
│   │   │   ├── mrc_tokenizer.py
│   │   │   ├── nlpc_wordseg_tokenizer.py
│   │   │   ├── tokenization_erniem.py
│   │   │   ├── tokenization_mix.py
│   │   │   ├── tokenization_spm.py
│   │   │   ├── tokenization_utils.py
│   │   │   ├── tokenization_wp.py
│   │   │   └── tokenizer.py
│   │   ├── util_helper.py
│   │   └── vocabulary.py
│   ├── __init__.py
│   ├── metrics
│   │   ├── chunk_metrics.py
│   │   ├── gen_eval.py
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── mrr.py
│   │   ├── tokenization.py
│   │   └── tuple.py
│   ├── model
│   │   ├── base_ernie_model.py
│   │   ├── __init__.py
│   │   └── model.py
│   ├── modules
│   │   ├── encoder.py
│   │   ├── ernie_config.py
│   │   ├── ernie_factory.py
│   │   ├── ernie_gen.py
│   │   ├── ernie_lr.py
│   │   ├── ernie.py
│   │   ├── __init__.py
│   │   ├── token_embedding
│   │   │   ├── base_token_embedding.py
│   │   │   ├── custom_fluid_embedding.py
│   │   │   ├── custom_token_embedding.py
│   │   │   └── __init__.py
│   │   └── transformer_encoder_gen.py
│   ├── utils
│   │   ├── args.py
│   │   ├── __init__.py
│   │   ├── log.py
│   │   ├── multi_process_eval.py
│   │   ├── params.py
│   │   ├── util_helper.py
│   │   └── visual_manager.py
│   └── version.py
└── readme.md
######其余模型项目结构如上
-|ernie_dqa_task1_ner_crf  
-|ernie_dqa_task1_seg_attention
-|ernie_dqa_task1_seg_attention_cls
-|combine.py
-README.MD
-main.ipynb #所有在AI Studio项目运行的顺序，基本上能够完成：数据处理->训练->推理->提交文件生成的所有流程
```
## 3.使用方式
> 本项目结构非常简单，使用方式可以直接启动本项目新的AI Studio项目副本，按照main.ipynb的指示即可完成所有操作
A：在AI Studio上[运行本项目](https://aistudio.baidu.com/aistudio/projectdetail/4950227)  
B：运行AISTUdio的副本，按照main.ipynb指示操作即可

## 4.参数设置、参数遍历
```
参数设置均为官方默认设置，epoch随着模型的优化，改为5，直接采用5个epoch,保存模型文件每1500step保存一次，直接使用附近提交多次，寻找最好的一次
模型作为最后的模型
```
