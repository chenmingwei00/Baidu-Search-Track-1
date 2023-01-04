# -*- coding: utf-8 -*
"""DyGraphTrainer
"""
import collections
import logging
import time
import paddle
from paddle.distributed import fleet
from erniekit.common.register import RegisterSet
from erniekit.common.rule import InstanceName
from erniekit.controller.dynamic_trainer import BaseDynamicTrainer


@RegisterSet.trainer.register
class CustomDynamicTrainer(BaseDynamicTrainer):
    """CustomDynamicTrainer
    """
    def __init__(self, params, data_set_reader, model):
        """
        :param params:
        :param data_set_reader:
        :param model_class:
        """
        BaseDynamicTrainer.__init__(self, params, data_set_reader, model)

    def do_train(self):
        """
        :return:
        """
        dg = self.data_set_reader.train_reader

        steps = 1
        opt_params = self.original_model.model_params.get('optimization', None)
        init_loss_scaling = opt_params.get("init_loss_scaling", 1.0)
        incr_every_n_steps = opt_params.get("incr_every_n_steps", 1000)
        decr_every_n_nan_or_inf = opt_params.get("decr_every_n_nan_or_inf", 2)
        incr_ratio = opt_params.get("incr_ratio", 2.0)
        decr_ratio = opt_params.get("decr_ratio", 0.8)

        if self.use_amp:
            self.scaler = paddle.amp.GradScaler(enable=self.use_amp,
                                                init_loss_scaling=init_loss_scaling,
                                                incr_ratio=incr_ratio,
                                                decr_ratio=decr_ratio,
                                                incr_every_n_steps=incr_every_n_steps,
                                                decr_every_n_nan_or_inf=decr_every_n_nan_or_inf)
            if self.multi_devices:
                self.scaler = fleet.distributed_scaler(self.scaler)

        time_begin = time.time()
        for batch_id, data in enumerate(dg()):
            self.model_class.train()
            with paddle.amp.auto_cast(enable=self.use_amp):
                data_ids, data_tokens = data
                example = self.data_set_reader.train_reader.dataset.convert_fields_to_dict(data_ids, need_emb=False)
               
                forward_out = self.model_class(example, phase=InstanceName.TRAINING)
                loss = forward_out[InstanceName.LOSS]
            if self.use_amp:
                loss = self.scaler.scale(loss)
                loss.backward()
                self.scaler.minimize(self.optimizer, loss)
            else:
                loss.backward()
                self.optimizer.minimize(loss)
                self.optimizer.step()
            self.model_class.clear_gradients()

            if self.original_model.lr_scheduler:
                cur_lr = self.original_model.lr_scheduler.get_lr()
                self.original_model.lr_scheduler.step()
            else:
                cur_lr = self.original_model.lr

            self.optimizer.clear_grad()

            if steps % self.params["train_log_step"] == 0:
                time_end = time.time()
                used_time = time_end - time_begin
                
                metrics_output = self.original_model.get_metrics(forward_out, data_tokens)
                logging.info("phase = {0} f1 = {1} precision = {2} recall = {3} step = {4} time_cost = {5}".format(
                    InstanceName.TRAINING, round(metrics_output['f1'], 3), round(metrics_output['precision'], 3), round(metrics_output['recall'], 3), steps, used_time))
                logging.info("current learning rate: {0}, loss: {1}".format(round(cur_lr, 7), round(loss.item(), 3)))
                time_begin = time.time()

            if steps % self.params["eval_step"] == 0:
                if self.params["is_eval_dev"]:
                    self.do_evaluate(self.data_set_reader.dev_reader, InstanceName.EVALUATE, steps)
                if self.params["is_eval_test"]:
                    self.do_evaluate(self.data_set_reader.test_reader, InstanceName.TEST, steps)

            if steps % self.params["save_model_step"] == 0 and self.worker_index == 0:
                self.save_models(steps, example)

            steps += 1

        if self.params["is_eval_dev"]:
            logging.info("Final evaluate result: ")
            self.do_evaluate(self.data_set_reader.dev_reader, InstanceName.EVALUATE, steps)
        if self.params["is_eval_test"]:
            logging.info("Final test result: ")
            self.do_evaluate(self.data_set_reader.test_reader, InstanceName.TEST, steps)

        if self.worker_index == 0:
            self.save_models(steps, example)

    def do_evaluate(self, reader, phase, step):
        """
        :param reader:
        :param phase:
        :param step:
        :return: loss
        """
        step = 0
        with paddle.no_grad():
            time_begin = time.time()
            # 先切换到eval模式
            self.model_class.eval()

            f1_scores = []
            recalls = []
            precisions = []
            fetch_output_dict = collections.OrderedDict()
            for batch_id, data in enumerate(reader()):
                step += 1
                data_ids, data_tokens = data
                example = reader.dataset.convert_fields_to_dict(data_ids, need_emb=False)
                forward_out = self.model_class(example, phase=phase)
                metrics_output = self.original_model.get_metrics(forward_out, data_tokens)
                f1_scores.append(metrics_output['f1'])
                recalls.append(metrics_output['recall'])
                precisions.append(metrics_output['precision'])

            time_end = time.time()
            used_time = time_end - time_begin
            self.model_class.train()
        logging.info("eval step = {0}".format(step))
        logging.info("phase = {0} f1 = {1} precision = {2} recall = {3} step = {4}, timecost = {5}".format(
                    phase, round(metrics_output['f1'], 3), round(metrics_output['precision'], 3), round(metrics_output['recall'], 3), step, used_time))

