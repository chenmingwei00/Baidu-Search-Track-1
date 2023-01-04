# -*- coding: utf-8 -*
"""
:py:class:`ErnieTextFieldReader`

"""
import paddle
# import logging
from paddle import fluid
from ...common.register import RegisterSet
from ...common.rule import DataShape, FieldLength, InstanceName
from .base_field_reader import BaseFieldReader
from ..util_helper import pad_batch_data, get_random_pos_id
# from wenxin.modules.token_embedding.ernie_embedding import ErnieTokenEmbedding
from ...utils.util_helper import truncation_words


@RegisterSet.field_reader.register
class ErnieTextFieldReader(BaseFieldReader):
    """使用ernie的文本类型的field_reader，用户不需要自己分词
        处理规则是：自动添加padding,mask,position,task,sentence,并返回length
        """
    def __init__(self, field_config):
        """
        :param field_config:
        """
        BaseFieldReader.__init__(self, field_config=field_config)

        if self.field_config.tokenizer_info:
            tokenizer_class = RegisterSet.tokenizer.__getitem__(self.field_config.tokenizer_info["type"])
            params = None
            if self.field_config.tokenizer_info.__contains__("params"):
                params = self.field_config.tokenizer_info["params"]
            self.tokenizer = tokenizer_class(vocab_file=self.field_config.vocab_path,
                                             split_char=self.field_config.tokenizer_info["split_char"],
                                             unk_token=self.field_config.tokenizer_info["unk_token"],
                                             params=params)
        self.label_map={'O': 0, 'B_ans': 1, 'I_ans': 2, 'E_ans': 3, 'X': 4, '[CLS]': 5, '[SEP]': 6}
        # logging.info("embedding_info = %s" % self.field_config.embedding_info)
        # if self.field_config.embedding_info and self.field_config.embedding_info["use_reader_emb"]:
        #     self.token_embedding = ErnieTokenEmbedding(emb_dim=self.field_config.embedding_info["emb_dim"],
        #                                                vocab_size=self.tokenizer.vocabulary.get_vocab_size(),
        #                                                params_path=self.field_config.embedding_info["config_path"])

    def init_reader(self, dataset_type=InstanceName.TYPE_PY_READER):
        """ 初始化reader格式，两种模式，如果是py_reader模式的话，返回reader的shape、type、level；
        如果是data_loader模式，返回fluid.data数组
        :param dataset_type : dataset的类型，目前有两种：py_reader、data_loader， 默认是py_reader
        :return:
        """
        #ToDo: 如果想使用静态图网络，需要修改此函数

        shape = []
        types = []
        levels = []
        feed_names = []
        data_list = []

        if self.field_config.data_type == DataShape.STRING:
            """src_ids"""
            shape.append([-1, -1])
            levels.append(0)
            types.append('int64')
            feed_names.append(self.field_config.name + "_" + InstanceName.SRC_IDS)
        else:
            raise TypeError("ErnieTextFieldReader's data_type must string")

        """sentence_ids"""
        shape.append([-1, -1])
        levels.append(0)
        types.append('int64')
        feed_names.append(self.field_config.name + "_" + InstanceName.SENTENCE_IDS)

        """position_ids"""
        shape.append([-1, -1])
        levels.append(0)
        types.append('int64')
        feed_names.append(self.field_config.name + "_" + InstanceName.POS_IDS)

        """mask_ids"""
        shape.append([-1, -1])
        levels.append(0)
        types.append('float32')
        feed_names.append(self.field_config.name + "_" + InstanceName.MASK_IDS)

        """task_ids"""
        shape.append([-1, -1])
        levels.append(0)
        types.append('int64')
        feed_names.append(self.field_config.name + "_" + InstanceName.TASK_IDS)

        """seq_lens"""
        shape.append([-1])
        levels.append(0)
        types.append('int64')
        feed_names.append(self.field_config.name + "_" + InstanceName.SEQ_LENS)

        "label start index"
        shape.append([-1, -1])
        levels.append(0)
        types.append('int64')
        feed_names.append(self.field_config.name + "_" + "label_start_index")

        "label end index"
        shape.append([-1, -1])
        levels.append(0)
        types.append('int64')
        feed_names.append(self.field_config.name + "_" + "label_end_index")

        "label ner index"
        shape.append([-1, -1])
        levels.append(0)
        types.append('int64')
        feed_names.append(self.field_config.name + "_" + "label_ner_index")


        if dataset_type == InstanceName.TYPE_DATA_LOADER:
            for i in range(len(feed_names)):
                data_list.append(paddle.static.data(name=feed_names[i], shape=shape[i],
                                                   dtype=types[i], lod_level=levels[i]))
            return data_list
        else:
            return shape, types, levels

    def convert_texts_to_ids(self, batch_text, use_random_pos=False, max_pos_id=2048):
        """将一个batch的明文text转成id
        :param batch_text:
        :return:
        """
        src_ids = []
        position_ids = []
        task_ids = []
        sentence_ids = []
        labels_start = []
        labels_end = []
        src_tokens = []
        org_answers = []
        answer_start_lists = []
        token_labels=[]
        for sample in batch_text:
            # if sample['answer_start_list']==[174]:
            #     print('111111111111111111')
            query = sample['query']
            doc = sample['document']
            org_answer = sample['org_answer']
            label=sample['label'].strip().split('\t')
            answer_list = sample['answer_list']
            answer_start_list = sample['answer_start_list']
            answer_start_lists.append(answer_start_list)

            tokens_org_answer = self.tokenizer.tokenize(org_answer)
            org_answers.append(tokens_org_answer)

            # tokens_query = self.tokenizer.tokenize(query)
            # tokens_doc = self.tokenizer.tokenize(doc)

            tokens_query=list(query)


            tokens_doc = list(doc)

            # 加上截断策略
            if len(tokens_doc) > self.field_config.max_seq_len - 3 - len(tokens_query):
                tokens_doc = truncation_words(tokens_doc, self.field_config.max_seq_len - 3 - len(tokens_query),
                                                self.field_config.truncation_type)
                label=label[:len(tokens_doc)]
            sentence_id = []
            tokens = []
            tokens.append("[CLS]")
            sentence_id.append(0)
            token_label=['O']

            for i, word in enumerate(tokens_query):
                # char seg for Chinese
                token = self.tokenizer.tokenize(word)
                tokens.extend(token)
                label_1 = 'O'
                for m in range(len(token)):
                    if m == 0:
                        token_label.append(label_1)
                    else:
                        token_label.append("X")
                    sentence_id.append(0)

            assert len(sentence_id)==len(token_label)
            assert len(sentence_id)==len(tokens)


            tokens.append("[SEP]")
            sentence_id.append(0)
            token_label.append('O')

            if len(label)>1: #说明没有标签，为预测阶段
                assert len(tokens_doc)==len(label)
            for i, word in enumerate(tokens_doc): #单纯截断只保证前面的answer上下文得到了训练，如果答案超过则无法保证训练
                # char seg for Chinese
                token = self.tokenizer.tokenize(word)
                tokens.extend(token)
                if len(label)>1:
                    label_1 = label[i]
                else:
                    label_1 = 'O'
                for m in range(len(token)):
                    if m == 0:
                        token_label.append(label_1)
                    else:
                        token_label.append("X")
                    sentence_id.append(1)

            # for token in tokens_doc:
            #     tokens.append(token)
            #     sentence_id.append(1)
            tokens.append("[SEP]")
            sentence_id.append(1)
            token_label.append('O')
            if 'B_ans' not in token_label and 'I_ans' in token_label: #说明B_ans位置可能为空
                start_=token_label.index('I_ans')
                token_label[start_]='B_ans'
            assert len(sentence_id) == len(token_label)
            assert len(sentence_id) == len(tokens)

            src_tokens.append(tokens)

            src_id = self.tokenizer.convert_tokens_to_ids(tokens)
            
            src_ids.append(src_id)

            token_label_ids=[self.label_map[ele] for ele in token_label]
            token_labels.append(token_label_ids)

            pos_id = list(range(len(src_id)))
            task_id = [0] * len(src_id)
            position_ids.append(pos_id)
            task_ids.append(task_id)
            sentence_ids.append(sentence_id)
            
            answer_span = []         
            label_start = [0.0] * len(src_id)
            label_end = [0.0] * len(src_id)
            if org_answer != "NoAnswer":
                for answer, answer_start in zip(answer_list, answer_start_list):
                    pre_doc = doc[:answer_start]
                    tokens_pre_doc=[]
                    for i, word in enumerate(pre_doc):  # 单纯截断只保证前面的answer上下文得到了训练，如果答案超过则无法保证训练
                        # char seg for Chinese
                        token = self.tokenizer.tokenize(word)
                        tokens_pre_doc.extend(token)
                    tokens_answer=[]
                    for i, word in enumerate(answer):  # 单纯截断只保证前面的answer上下文得到了训练，如果答案超过则无法保证训练
                        # char seg for Chinese
                        token = self.tokenizer.tokenize(word)
                        tokens_answer.extend(token)

                    # tokens_pre_doc = self.tokenizer.tokenize(pre_doc)
                    # tokens_answer = self.tokenizer.tokenize(answer)

                    start_index = len(tokens_pre_doc) + len(tokens_query) + 2 if answer_start != 0 else len(tokens_query) + 2
                    end_index = start_index + len(tokens_answer) - 1
                    answer_span.append((start_index, end_index))
            else:
                answer_span.append((0, 0))
            for span in answer_span:  #此处为token之后的答案所在start和end位置
                if span[0] < len(label_start):
                    label_start[span[0]] = 1.0
                if span[1] < len(label_end): 
                    label_end[span[1]] = 1.0
            labels_start.append(label_start)
            labels_end.append(label_end)

        return_list_ids = []
        return_list_tokens = []
        padded_ids, input_mask, batch_seq_lens = pad_batch_data(src_ids,
                                                                pad_idx=self.field_config.padding_id,
                                                                return_input_mask=True,
                                                                return_seq_lens=True)
        sent_ids_batch = pad_batch_data(sentence_ids, pad_idx=self.field_config.padding_id)
        pos_ids_batch = pad_batch_data(position_ids, pad_idx=self.field_config.padding_id)
        task_ids_batch = pad_batch_data(task_ids, pad_idx=self.field_config.padding_id)
        token_labels_batch = pad_batch_data(token_labels, pad_idx=self.label_map['O'])


        return_list_ids.append(padded_ids)  # append src_ids
        return_list_ids.append(sent_ids_batch)  # append sent_ids
        return_list_ids.append(pos_ids_batch)  # append pos_ids
        return_list_ids.append(input_mask)  # append mask_ids
        return_list_ids.append(task_ids_batch)  # append task_ids
        return_list_ids.append(batch_seq_lens)  # append seq_lens
        return_list_tokens.append(src_tokens)   # src_tokens
        return_list_tokens.append(org_answers)  # org_answers

        labels_start_batch = pad_batch_data(labels_start, pad_idx=-100)
        labels_end_batch = pad_batch_data(labels_end, pad_idx=-100)
        return_list_ids.append(labels_start_batch)   # start index
        return_list_ids.append(labels_end_batch)     # end index
        return_list_ids.append(token_labels_batch)
        return return_list_ids, return_list_tokens

    def structure_fields_dict(self, fields_id, start_index, need_emb=True):
        """静态图调用的方法，生成一个dict， dict有两个key:id , emb. id对应的是pyreader读出来的各个field产出的id，
        emb对应的是各个field对应的embedding
        :param fields_id: pyreader输出的完整的id序列
        :param start_index:当前需要处理的field在field_id_list中的起始位置
        :param need_emb:是否需要embedding（预测过程中是不需要embedding的）
        :return:
        """
        record_id_dict = {}
        record_id_dict['src_ids'] = fields_id[start_index]
        record_id_dict['sent_ids'] = fields_id[start_index + 1]
        record_id_dict['pos_ids'] = fields_id[start_index + 2]
        record_id_dict['mask_ids'] = fields_id[start_index + 3]
        record_id_dict['task_ids'] = fields_id[start_index + 4]
        record_id_dict['seq_lens'] = fields_id[start_index + 5]
        record_id_dict['label_start_index'] = fields_id[start_index + 6]
        record_id_dict['label_end_index'] = fields_id[start_index + 7]
        record_id_dict['label_ner_index'] = fields_id[start_index + 8]

        record_emb_dict = None
        if need_emb and self.token_embedding:
            record_emb_dict = self.token_embedding.get_token_embedding(record_id_dict)

        record_dict = {}
        record_dict[InstanceName.RECORD_ID] = record_id_dict
        record_dict[InstanceName.RECORD_EMB] = record_emb_dict

        return record_dict

    def get_field_length(self):
        """获取当前这个field在进行了序列化之后，在field_id_list中占多少长度
        :return:
        """
        return FieldLength.ERNIE_TEXT_FIELD + 2    # start index and end index


