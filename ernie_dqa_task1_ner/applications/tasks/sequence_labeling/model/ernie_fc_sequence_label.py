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
                                              weight_attr=paddle.ParamAttr(name='cls_seq_ner_label_out_w', initializer=initializer),
                                              bias_attr='cls_seq_ner_label_out_b')
        # self.loss = paddle.nn.CrossEntropyLoss(use_softmax=False, reduction='none')

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
        # tokens_embedding = [batch_size, max_seq_len, hidden_size]
        _, tokens_embedding = self.ernie_model(src_ids=text_src, sent_ids=text_sent,
                                                           task_ids=text_task)

        emb_text = self.dropout(tokens_embedding)

        # [batch_size, max_seq_len] 
        logits_start = self.fc_prediction_start(emb_text)
        logits_end = self.fc_prediction_end(emb_text)
        logits_ner = self.fc_prediction_ner(emb_text)
        probs_start = paddle.squeeze(paddle.nn.functional.sigmoid(logits_start), axis=2)
        probs_end = paddle.squeeze(paddle.nn.functional.sigmoid(logits_end), axis=2)

        #ner loss caculate
        log_probs = paddle.nn.functional.softmax(logits_ner,axis=-1)

        # log_probs = paddle.nn.functional.log_softmax(logits_ner, axis=-1)
        one_hot_labels = paddle.nn.functional.one_hot(label_ner, num_classes=self.number_ner)
        # per_example_loss = -paddle.fluid.layers.reduce_sum(one_hot_labels * log_probs, dim=-1)
        # loss_ner = paddle.fluid.layers.reduce_sum(per_example_loss)

        # loss_start = self.compute_focal_loss(probs_start, label_start)
        # loss_end = self.compute_focal_loss(probs_end, label_end)
        loss_start = self.compute_focal_loss(probs_start, label_start, text_mask)
        loss_end = self.compute_focal_loss(probs_end, label_end, text_mask)
        text_mask_ner=paddle.unsqueeze(text_mask,axis=-1)

        loss_ner = self.compute_focal_loss(log_probs, one_hot_labels, text_mask_ner)

        loss = (loss_start + loss_end+loss_ner) / 3.0

        probs_start = probs_start * text_mask
        probs_end = probs_end * text_mask
        probs_ner = log_probs * text_mask_ner

        if phase == InstanceName.SAVE_INFERENCE:
            target_predict_list = [probs_start, probs_end, probs_ner]
            target_feed_list = [text_src, text_sent, text_mask]
            target_feed_name_list = ["text_a#src_ids", "text_a#sent_ids", "text_a#mask_ids"]

            forward_return_dict = {
                InstanceName.TARGET_FEED: target_feed_list,
                InstanceName.TARGET_FEED_NAMES: target_feed_name_list,
                InstanceName.TARGET_PREDICTS: target_predict_list
            }
            return forward_return_dict

        forward_return_dict = {
            "length": text_lens,
            "loss": loss,
            "probs_start": probs_start,
            "probs_end": probs_end,
            "probs_ner": probs_ner,
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
