# -*- coding: utf-8 -*
"""
ErnieFcSeqLabel
"""
import sys
sys.path.append("../../../")
import paddle
import re
from erniekit.common.register import RegisterSet
from erniekit.common.rule import InstanceName
from erniekit.modules.ernie import ErnieModel
from erniekit.model.model import BaseModel
from erniekit.modules.ernie_config import ErnieConfig
from erniekit.modules.ernie_lr import LinearWarmupDecay
from erniekit.metrics import chunk_metrics
import logging
import collections
from paddle import fluid


class BLSTM_CRF(object):
    def __init__(self,embedding,hidden_unit,cell_type,num_layer,
                    dropout_rate,max_seq_length,labes,length,training):
        """
        BLSTM-CRF 网络
        :param embedded_chars: Fine-tuning embedding input
        :param hidden_unit: LSTM的隐含单元个数
        :param cell_type: RNN类型（LSTM OR GRU DICNN will be add in feature）
        :param num_layers: RNN的层数
        :param droupout_rate: droupout rate
        :param initializers: variable init class
        :param num_labels: 标签数量
        :param seq_length: 序列最大长度
        :param labels: 真实标签
        :param lengths: [batch_size] 每个batch下序列的真实长度
        :param is_training: 是否是训练过程
        """
        self.hidden_unit = hidden_unit
        self.dropout_rate = dropout_rate
        self.cell_type = cell_type
        self.num_layers = num_layer
        self.embedded_chars = embedding
        self.max_seq_length = max_seq_length
        self.num_labels = 7
        self.labels = labes
        self.lengths = length
        self.embedding_dims = embedding.shape[-1]
        self.is_traing=training

        initializer = paddle.nn.initializer.TruncatedNormal(std=0.02)
        self.project_bilstm_layer = paddle.nn.Linear(in_features=hidden_unit*2, out_features=self.num_labels,
                                                  weight_attr=paddle.ParamAttr(name='cls_seq_ner_label',
                                                                               initializer=initializer),
                                                  bias_attr='cls_seq_ner_label_out')

    def add_blstm_crf_layer(self, crf_only,logits_ner):
        """
        blstm-crf网络
        :return:
        """
        # if self.is_training:
        #     # lstm input dropout rate i set 0.9 will get best score
        #     self.embedded_chars = tf.nn.dropout(self.embedded_chars, self.dropout_rate)

        if crf_only:
            logits = logits_ner
        else:
            # blstm
            lstm_output = self.blstm_layer(self.embedded_chars)
            # project
            logits = self.project_bilstm_layer(lstm_output)
        # crf
        loss,crf_decode = self.crf_layer_loss(logits,self.labels,self.lengths)
        loss /= 100
        # CRF decode, pred_ids 是一条最大概率的标注路径
        return (loss, logits,crf_decode)

    def _witch_cell(self):
        """
        RNN 类型
        :return:
        """
        cell_tmp = None
        if self.cell_type == 'lstm':
            cell_tmp = paddle.nn.LSTM(self.embedding_dims,self.hidden_unit, num_layers=self.num_layers,
                                       direction='bidirect', dropout=self.dropout_rate)
        elif self.cell_type == 'gru':
            cell_tmp = paddle.nn.LSTM.GRUCell(self.embedding_dims,self.hidden_unit, num_layers=self.num_layers,
                                       direction='bidirect', dropout=self.dropout_rate)
        return cell_tmp

    def _bi_dir_rnn(self):
        """
        双向RNN
        :return:
        """
        cell_fw = self._witch_cell()
        # if self.dropout_rate is not None:
        #     cell_bw = rnn.DropoutWrapper(cell_bw, output_keep_prob=self.dropout_rate)
        #     cell_fw = rnn.DropoutWrapper(cell_fw, output_keep_prob=self.dropout_rate)
        return cell_fw

    def blstm_layer(self, embedding_chars):
        """

        :return:
        """
        self.lstm=self._bi_dir_rnn()
        outputs, (hidden, cell) = self.lstm(self.embedded_chars,sequence_length=self.lengths)  # [batch_size,time_steps,num_directions * hidden_size]

        return outputs

    def project_crf_layer(self, embedding_chars, name=None):
        """
        hidden layer between input layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project" if not name else name):
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.embedding_dims, self.num_labels],
                                    dtype=tf.float32, initializer=self.initializers.xavier_initializer())

                b = tf.get_variable("b", shape=[self.num_labels], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(self.embedded_chars,
                                    shape=[-1, self.embedding_dims])  # [batch_size, embedding_dims]
                pred = tf.tanh(tf.nn.xw_plus_b(output, W, b))
            return tf.reshape(pred, [-1, self.seq_length, self.num_labels])

    def crf_layer_loss(self, feature_out,target,length):
        """
        calculate crf loss
        :param project_logits: [1, num_steps, num_tags]
        :return: scalar loss
        """
        length=paddle.cast(paddle.unsqueeze(length,axis=-1),dtype=paddle.int64)
        target=paddle.unsqueeze(target,axis=-1)
        # 调用内置 CRF 函数并做状态转换解码.
        crf_cost2 = fluid.layers.linear_chain_crf(
            input=feature_out,
            label=target,
            length=length,
            param_attr=fluid.ParamAttr(
                name='crfw',
                learning_rate=2e-5))

        crf_decode = paddle.static.nn.crf_decoding(input=feature_out,
                                                   length=length,
                                                   param_attr=paddle.ParamAttr(name="crfw")
                                                   )

        avg_cost = fluid.layers.mean(crf_cost2)

        return avg_cost,crf_decode
import math
@RegisterSet.models.register
class ErnieFcSeqLabel(BaseModel):
    """ErnieMatchingFcPointwise:使用TextFieldReader组装数据,只返回src_id和length，用户可以使用src_id自己生成embedding
    """

    def __init__(self, model_params):
        BaseModel.__init__(self, model_params)
        self.config_path = self.model_params["embedding"].get("config_path")
        self.number_ner=8
    def structure(self):
        """网络结构组织
        :return:
        """
        # self.num_labels = self.model_params.get('num_labels', 7)
        self.cfg_dict = ErnieConfig(self.config_path)
        self.hid_dim = self.cfg_dict['hidden_size']

        self.ernie_model = ErnieModel(self.cfg_dict, name='')
        initializer = paddle.nn.initializer.TruncatedNormal(std=0.02)
        self.dropout = paddle.nn.Dropout(p=0.1, mode="upscale_in_train")
        # self.fc_prediction = paddle.nn.Linear(in_features=self.hid_dim, out_features=self.num_labels,
                                            #   weight_attr=paddle.ParamAttr(name='cls_seq_label_out_w',
                                                                        #    initializer=initializer),
                                            #   bias_attr='cls_seq_label_out_b')
        # self.fc = paddle.nn.Linear(in_features=self.num_labels, out_features=self.num_labels)
        self.fc_prediction_start = paddle.nn.Linear(in_features=self.hid_dim, out_features=1,
                                              weight_attr=paddle.ParamAttr(name='cls_seq_start_label_out_w', initializer=initializer),
                                              bias_attr='cls_seq_start_label_out_b')
        self.fc_prediction_end = paddle.nn.Linear(in_features=self.hid_dim, out_features=1,
                                              weight_attr=paddle.ParamAttr(name='cls_seq_end_label_out_w', initializer=initializer),
                                              bias_attr='cls_seq_end_label_out_b')

        self.fc_prediction_ner = paddle.nn.Linear(in_features=self.hid_dim, out_features=self.number_ner,
                                                  weight_attr=paddle.ParamAttr(name='cls_seq_ner_label_out_w',
                                                                               initializer=initializer),
                                                  bias_attr='cls_seq_ner_label_out_b')

        self.key_layer = paddle.nn.Linear(in_features=self.hid_dim, out_features=self.hid_dim,
                                          weight_attr=paddle.ParamAttr(name='key_out_w',
                                                                       initializer=initializer),
                                          bias_attr='key_out_b')

        self.query_layer = paddle.nn.Linear(in_features=self.hid_dim, out_features=self.hid_dim,
                                            weight_attr=paddle.ParamAttr(name='query_out_w',
                                                                         initializer=initializer),
                                            bias_attr='query_out_b')

        self.value_layer = paddle.nn.Linear(in_features=self.hid_dim, out_features=self.hid_dim,
                                            weight_attr=paddle.ParamAttr(name='value_out_w',
                                                                         initializer=initializer),
                                            bias_attr='value_out_b')
    def net_forward(self,
                    text_src,
                    text_sent,
                    text_task,
                    label_start,
                    label_end,
                    text_mask,
                    label_ner,
                    first_embedding=None):

        # tokens_embedding = [batch_size, max_seq_len, hidden_size]
        cls_embed, tokens_embedding = self.ernie_model(src_ids=text_src, sent_ids=text_sent,
                                               task_ids=text_task)

        emb_text = self.dropout(tokens_embedding)

        if first_embedding:
            # pre attetion
            self.key_embed = paddle.unsqueeze(self.key_layer(first_embedding), axis=-1)
            self.query_embed = self.query_layer(emb_text)
            self.value_embed = self.value_layer(emb_text)

            attention_scores = paddle.matmul(self.query_embed, self.key_embed)
            attention_scores = attention_scores * 1.0 / math.sqrt(self.hid_dim)
            value_embed_new = paddle.multiply(self.value_embed, attention_scores)

            emb_text = self.dropout(value_embed_new)

        # [batch_size, max_seq_len]
        logits_start = self.fc_prediction_start(emb_text)
        logits_end = self.fc_prediction_end(emb_text)
        probs_start = paddle.squeeze(paddle.nn.functional.sigmoid(logits_start), axis=2)
        probs_end = paddle.squeeze(paddle.nn.functional.sigmoid(logits_end), axis=2)

        # ner loss caculate
        logits_ner = self.fc_prediction_ner(emb_text)  # 可以理解为发射概率

        max_seq_length = text_src.shape[-1]
        # 计算序列的真实长度
        used = paddle.sign(paddle.abs(paddle.cast(text_src, paddle.float32)))
        seq_length = paddle.cast(paddle.fluid.layers.reduce_sum(used, dim=-1), dtype=paddle.int32)
        blstm_crf = BLSTM_CRF(embedding=emb_text, hidden_unit=128, cell_type='lstm', num_layer=1,
                              dropout_rate=0.1, max_seq_length=max_seq_length, labes=label_ner, length=seq_length,
                              training=self.training)
        loss_ner, log_probs, crf_probs = blstm_crf.add_blstm_crf_layer(crf_only=True, logits_ner=logits_ner)

        loss_start = self.compute_focal_loss(probs_start, label_start, text_mask)
        loss_end = self.compute_focal_loss(probs_end, label_end, text_mask)

        text_mask_ner = paddle.unsqueeze(text_mask, axis=-1)

        loss = (loss_start + loss_end + loss_ner) / 3.0
        probs_start = probs_start * text_mask
        probs_end = probs_end * text_mask
        probs_ner = log_probs * text_mask_ner
        probs_crf = crf_probs * text_mask

        return loss,probs_start,probs_end,probs_ner,probs_crf,cls_embed

    def forward(self, fields_dict, phase):
        """前向计算组网部分，必须由子类实现
        :return: loss , fetch_list
        """
        instance_text = fields_dict["text_a"]
        record_id_text = instance_text[InstanceName.RECORD_ID]
        input_dict = fields_dict['text_a']['id']
        text_src = input_dict['src_ids']
        text_sent = input_dict['sent_ids']
        text_mask = input_dict['mask_ids']
        text_task = input_dict['task_ids']
        # [batch_size]
        text_lens = input_dict['seq_lens']
        # [batch_size, max_seq_len]
        label_start = input_dict['label_start_index']
        label_end = input_dict['label_end_index']
        label_ner = input_dict['label_ner_index']
        loss1, probs_start1, probs_end1, probs_ner1, probs_crf1,cls_embed=self.net_forward(text_src,
                                                                                 text_sent,
                                                                                 text_task,
                                                                                 label_start,
                                                                                 label_end,
                                                                                 text_mask,
                                                                                 label_ner)

        #第二段的预测结果
        text_src2 = input_dict['src_ids2']
        text_sent2 = input_dict['sent_ids2']
        text_mask2 = input_dict['mask_ids2']
        text_task2 = input_dict['task_ids2']
        # [batch_size]
        text_lens2 = input_dict['seq_lens2']
        # [batch_size, max_seq_len]
        label_start2 = input_dict['label_start_index2']
        label_end2 = input_dict['label_end_index2']
        label_ner2 = input_dict['label_ner_index2']
        # tokens_embedding = [batch_size, max_seq_len, hidden_size]

        loss2, probs_start2, probs_end2, probs_ner2,probs_crf2,_ = self.net_forward(
                                                                           text_src2,
                                                                           text_sent2,
                                                                           text_task2,
                                                                           label_start2,
                                                                           label_end2,
                                                                           text_mask2,
                                                                           label_ner2,
                                                                            cls_embed,
                                                                            )


        loss=loss1+loss2

        if phase == InstanceName.SAVE_INFERENCE:
            target_predict_list = [probs_start1, probs_end1, probs_ner1,probs_crf1,
                                   probs_start2, probs_end2, probs_ner2, probs_crf2]
            target_feed_list = [text_src, text_sent, text_mask, text_src2, text_sent2, text_mask2]
            target_feed_name_list = ["text_a#src_ids", "text_a#sent_ids", "text_a#mask_ids",
                                     "text_a#src_ids2", "text_a#sent_ids2", "text_a#mask_ids2"]

            forward_return_dict = {
                InstanceName.TARGET_FEED: target_feed_list,
                InstanceName.TARGET_FEED_NAMES: target_feed_name_list,
                InstanceName.TARGET_PREDICTS: target_predict_list
            }
            return forward_return_dict

        forward_return_dict = {
            "length": text_lens,
            "loss": loss,
            "probs_start": probs_start1,
            "probs_end": probs_end1,
            "probs_ner": probs_ner1,
            "probs_crf":probs_crf1,
            "probs_start2": probs_start2,
            "probs_end2": probs_end2,
            "probs_ner2": probs_ner2,
            "probs_crf2": probs_crf2,
        }
        return forward_return_dict

    def compute_focal_loss(self, probs, labels, mask, alpha=1.0, gamma=2.0):
        focal_loss = - ((1 - probs)**gamma * labels * paddle.log(probs) +  # positive samples
                alpha * (probs**gamma) * (1 - labels) * paddle.log(1 - probs))  # negative samples
        focal_loss = focal_loss * mask
        focal_loss = paddle.sum(focal_loss, axis=1)
        focal_loss = paddle.mean(x=focal_loss)
        return focal_loss

    def compute_bi_loss(self, probs, labels, mask, alpha=0.01):
        loss = - (labels * paddle.log(probs) + alpha * (1 - labels) * paddle.log(1 - probs))
        loss = loss * mask
        loss = paddle.sum(loss, axis=1)
        loss = paddle.mean(x=loss)
        return loss
    def set_optimizer(self):
        """
        :return: optimizer
        """
        # 学习率和权重的衰减设置在optimizer中，loss的缩放设置在amp中（各个trainer中进行设置）。
        # TODO:需要考虑学习率衰减、权重衰减设置、 loss的缩放设置
        opt_param = self.model_params.get('optimization', None)
        self.lr = opt_param.get("learning_rate", 2e-5)
        weight_decay = opt_param.get("weight_decay", 0.01)
        use_lr_decay = opt_param.get("use_lr_decay", False)
        epsilon = opt_param.get("epsilon", 1e-6)
        g_clip = paddle.nn.ClipGradByGlobalNorm(1.0)
        param_name_to_exclue_from_weight_decay = re.compile(r'.*layer_norm_scale|.*layer_norm_bias|.*b_0')

        parameters = None
        if self.is_dygraph:
            parameters = self.parameters()

        if use_lr_decay:
            max_train_steps = opt_param.get("max_train_steps", 0)
            warmup_steps = opt_param.get("warmup_steps", 0)
            self.lr_scheduler = LinearWarmupDecay(base_lr=self.lr, end_lr=0.0, warmup_steps=warmup_steps,
                                                  decay_steps=max_train_steps, num_train_steps=max_train_steps)
            self.optimizer = paddle.optimizer.AdamW(learning_rate=self.lr_scheduler,
                                                    parameters=parameters,
                                                    weight_decay=weight_decay,
                                                    apply_decay_param_fun=lambda
                                                        n: not param_name_to_exclue_from_weight_decay.match(n),
                                                    epsilon=epsilon,
                                                    grad_clip=g_clip)
        else:
            self.optimizer = paddle.optimizer.AdamW(self.lr,
                                                    parameters=parameters,
                                                    weight_decay=weight_decay,
                                                    apply_decay_param_fun=lambda
                                                        n: not param_name_to_exclue_from_weight_decay.match(n),
                                                    epsilon=epsilon,
                                                    grad_clip=g_clip)
        return self.optimizer
    # def back_backend(self,loss,startup_program,parameter_list):
    #
    #     params_grads = self.backward(loss,
    #                                  startup_program=startup_program,
    #                                  parameters=parameter_list,
    #                                  no_grad_set=None)
    #     return params_grads

