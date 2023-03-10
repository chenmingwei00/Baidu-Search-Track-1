# -*- coding: utf-8 -*
"""
对内工具包（major）中最常用的inference，必须继承自文心core中的BaseInference基类，必须实现inference_batch, inference_query方法。
"""
import logging
import os
import time
import numpy as np

from erniekit.common.register import RegisterSet
from erniekit.common.rule import InstanceName
from erniekit.controller.inference import BaseInference


@RegisterSet.inference.register
class CustomInference(BaseInference):
    """CustomInference
    """
    def __init__(self, params, data_set_reader, parser_handler):
        """
        :param params:前端json中设置的参数
        :param data_set_reader: 预测集reader
        :param parser_handler: 飞桨预测结果通过parser_handler参数回调到具体的任务中，由用户控制具体结果解析
        """
        BaseInference.__init__(self, params, data_set_reader, parser_handler)
    
    def inference_batch(self):
        """
        批量预测
        """
        logging.info("start do inference....")
        total_time = 0
        output_path = self.params.get("output_path", None)
        if not output_path or output_path == "":
            if not os.path.exists("./output"):
                os.makedirs("./output")
            output_path = "./output/predict_result.txt"

        output_file = open(output_path, "w+")

        dg = self.data_set_reader.predict_reader
        dg.dataset.need_generate_examples = True
        sample_entity_list = None
        with open('./output/40001.txt','w',encoding='utf-8') as ner_pre:
            for batch_id, data in enumerate(dg()):
                data_input, sample_entity_list = data
                data_ids, data_tokens = data_input
                feed_dict = dg.dataset.convert_fields_to_dict(data_ids)
                predict_results = []

                for index, item in enumerate(self.input_keys):
                    kv = item.split("#")
                    name = kv[0]
                    key = kv[1]
                    item_instance = feed_dict[name]
                    input_item = item_instance[InstanceName.RECORD_ID][key]
                    # input_item是tensor类型
                    self.input_handles[index].copy_from_cpu(input_item.numpy())
                begin_time = time.time()
                self.predictor.run()
                end_time = time.time()
                total_time += end_time - begin_time

                output_names = self.predictor.get_output_names()
                for i in range(len(output_names)):
                    output_tensor = self.predictor.get_output_handle(output_names[i])
                    predict_results.append(output_tensor)

                seq_lens = feed_dict["text_a"][InstanceName.RECORD_ID]["seq_lens"].numpy()
                # 回调给解析函数
                write_result_list, write_score_list,write_ner_results = self.parser_handler(predict_results[:4],
                        input_list=data_tokens[0], params_dict=self.params, seq_lens=seq_lens)

                write_result_list2, write_score_list2, write_ner_results2 = self.parser_handler(predict_results[4:],
                                                                                             input_list=data_tokens[2],
                                                                                             params_dict=self.params,
                                                                                             seq_lens=seq_lens)
                for score, result in zip(write_score_list, write_result_list):
                    output_file.write(str(score) + '\t' + result + '\n')

                for t_ner_pre,t_ner_pre2 in zip(write_ner_results,write_ner_results2):
                    score1,answer1=t_ner_pre.strip().split('\t')
                    score2,answer2=t_ner_pre2.strip().split('\t')
                    if answer1=='NoAnswer' and answer2=='NoAnswer':
                        ner_pre.write(t_ner_pre)
                    elif answer1!='NoAnswer' and answer2=='NoAnswer':
                        ner_pre.write(t_ner_pre)
                    elif answer1=='NoAnswer' and answer2!='NoAnswer':
                        ner_pre.write(t_ner_pre2)
                    else:
                        answer=answer1+'.'+answer2
                        score=(float(score1)+float(score2))/2.0
                        line=str(score)+'\t'+answer+'\n'
                        ner_pre.write(line)

        logging.info("total_time:{}".format(total_time))
        output_file.close()

    def inference_query(self, query):
        """单条query预测
        :param query
        """
        total_time = 0
        reader = self.data_set_reader.predict_reader.dataset
        reader.need_generate_examples = True
        data, sample = reader.api_generator(query)
        feed_dict = reader.convert_fields_to_dict(data)
        predict_results = []

        for index, item in enumerate(self.input_keys):
            kv = item.split("#")
            name = kv[0]
            key = kv[1]
            item_instance = feed_dict[name]
            input_item = item_instance[InstanceName.RECORD_ID][key]
            self.input_handles[index].copy_from_cpu(np.array(input_item))

        begin_time = time.time()
        self.predictor.run()
        end_time = time.time()
        total_time += end_time - begin_time

        output_names = self.predictor.get_output_names()
        for i in range(len(output_names)):
            output_tensor = self.predictor.get_output_handle(output_names[i])
            predict_results.append(output_tensor)

        # seq_lens = feed_dict["text_a"][InstanceName.RECORD_ID]["seq_lens"].numpy()
        seq_lens = feed_dict["text_a"][InstanceName.RECORD_ID]["seq_lens"]
        # 回调给解析函数
        result_list = self.parser_handler(predict_results,
                                                sample_list=sample, params_dict=self.params,
                                                seq_lens=seq_lens)
        return result_list
