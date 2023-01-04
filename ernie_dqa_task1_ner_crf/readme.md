进入ernie_dqa_task1/applications/tasks/sequence_labeling目录
train: sh begin_train.sh
inference: python run_infer.py --param_path=./examples/seqlab_ernie_fc_ch_infer.json
配置文件位于examples目录下
训练好的模型参数保存在output目录下
evaluation: sh evaluate.sh


base模型核心就是对于seq——len 预测句子长度的start位置二分类以及end句子长二分类，最后和实际句子答案位置的
start  +end 标签求损失
同时做了偏重正样本损失的结果，非start和end的损失降低权重

1改进1，对query 以及doc对应去除停用词后相同词语进行特征标注
标注方式1
    中国面积大小  中华人民共和国面积：陆地面积约960万平方公里，水域面积约470多万平方公里
    1 1 22                 2 2     2 2                  22
                         中华人民共和国(the People's Republic of China)，简称“中国”，成立于1949年10月1日，位于亚洲东部，太平洋西岸，...详情 >
                                                                          2 2
标注方式2
    <a>中国</a><a>面积</a>大小  中华人民共和国<a>面积</a>：陆地面积约960万平方公里，水域面积约470多万平方公里
                         中华人民共和国(the People's Republic of China)，简称“中国”，成立于1949年10月1日，位于亚洲东部，太平洋西岸，...详情 >
                                                                          2 2
                                                                          
 注解，ori_answer如果没有答案则是Noanswer而不是''                                                                   

1. 
   a1检查一下query的长度分布以及doc的分布情况，doc+query超过512的比例
   答案占512 后半部分的数据有多少
    经过统计发现，len(query)+len(doc_text)超过512长度占比 0.5540324642126789

    答案末尾长度超过512 竟然占0.44
    
    经分析doc——text长度分布比较均匀，所以一定要仔细处理doc的长度，而不是简单的截断
    
    (1) 
       首先对预测文本doc_text,按照最大文本512 进行分割，分别预测，最后答案进行合并
       /media/chen/T7/projects/baidu_dureare_qa_2task/ernie_dqa_task1/applications/
       tasks/sequence_labeling/data/data_deal.py
       原始test文本3868个样本
       
       a. test_data_split函数
          主要是把doctext按照512分割，以id为同一个文档为唯一标识，为了后来把同一文档答案合并做唯一标志
          分割后9631个样本
       
       b. 按照正常的预测模型进行推理得到最后的text答案 4000.txt 行数为顺序的9631个样本
       
       c. predict_deal函数 把结果按照id对答案进行合并最终在原始模型分数为0.61109提升为0.62846,训练步数为4000步0.628，
       充分说明如何处理超过512的长文本是至关重要的，不能简单的截断.
    (2)
        继承步骤（1）虽然这种强制截断有损失，但是还有一定的提高，如果训练数据同样的方法纪念性操作，获取预测相同的
        数据分布，应该有所提升，这一步训练数据采用相同的方式进行处理.实际效果很差，就舍弃了，因为信息截断对模型效果影响非常大
        
    (3)
       https://mbd.baidu.com/ug_share/mbox/4a83aa9e65/share?product=smartapp&tk=8683a38286cc2615e643f2
       64d7760faf&share_url=https%3A%2F%2Fyebd1h.smartapps.cn%2Fpages%2Fblog%2Findex%3FblogId%3D11217762
       1%26_swebfr%3D1%26_swebFromHost%3Dbaiduboxapp&domain=mbd.baidu.com
       这个连接方法的尝试,对多个分段试用前n-1segment段落的信息加入到n时刻段落信息
       
        1.首先查看与输入多个分段的位置有哪些
    (4)
       使用命名实体识别方法
       a.构造B-ans I-ans E-ans  O对应的标签训练数据
    (5)
       阅读理解的大模型尝试