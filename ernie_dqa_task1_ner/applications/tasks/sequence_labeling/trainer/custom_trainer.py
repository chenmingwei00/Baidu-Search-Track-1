# -*- coding: utf-8 -*
"""
对内工具包（major）中最常用的trainer，必须继承自文心core中的BaseTrainer基类，必须实现do_train, do_evaluate, do_visual方法。
"""
import collections
import logging
import time
import numpy as np
from erniekit.common.register import RegisterSet
from erniekit.common.rule import InstanceName
import paddle.distributed.fleet as fleet
from erniekit.controller.static_trainer import BaseStaticTrainer

@RegisterSet.trainer.register
class CustomTrainer(BaseStaticTrainer):
    """CustomTrainer
    """
    def __init__(self, params, data_set_reader, model):
        """
        :param params:前端json中设置的参数
        :param data_set_reader: 数据集实例，包括训练集、评估集、测试集、预测集
        :param model:模型组网实例
        """
        BaseStaticTrainer.__init__(self, params, data_set_reader, model)
        self.return_numpy = self.params.get("return_numpy", True)

    def do_train(self):
        """ 启动数据集循环，开始训练
        :return:
        """
        if self.use_fleet and fleet.is_server():
            logging.debug("is fleet.server, over")
            return
        if self.use_fleet:
            logging.debug("worker_index%d start train...." % fleet.worker_index())

        num_train_examples = self.params.get("num_train_examples", 0)
        if num_train_examples == 0:
            num_train_examples = self.data_set_reader.train_reader.get_num_examples()

        dg = self.data_set_reader.train_reader
        steps = 1
        time_begin = time.time()
        for batch_id, data in enumerate(dg()):
            if len(data) == 0:
                continue
            data_ids, data_tokens = data
            feed_dict = self.data_set_reader.train_reader.dataset.convert_input_list_to_dict(data_ids)
            if steps % self.params["train_log_step"] != 0:
                if self.use_fleet:
                    self.train_exe.run(program=self.train_program, feed=feed_dict, fetch_list=[], return_numpy=True)
                else:
                    self.train_exe.run(feed=feed_dict, fetch_list=[], return_numpy=True)
            else:
                if self.use_fleet:
                    fetch_output = self.train_exe.run(program=self.train_program,
                                                      feed=feed_dict,
                                                      fetch_list=self.fetch_list_train,
                                                      return_numpy=True)
                else:
                    fetch_output = self.train_exe.run(feed=feed_dict,
                                                      fetch_list=self.fetch_list_train,
                                                      return_numpy=True)

                current_example, current_epoch = self.data_set_reader.train_reader.dataset.get_train_progress()
                logging.info("epoch {0} progress {1}/{2}".format(current_epoch, current_example, num_train_examples))

                fetch_output_dict = collections.OrderedDict()
                for key, value in zip(self.fetch_list_train_key, fetch_output):
                    if key == InstanceName.LOSS and not self.return_numpy:
                        value = np.array(value)
                    fetch_output_dict[key] = value
                time_end = time.time()
                used_time = time_end - time_begin

                logging.info("phase = {0} step = {1} time_cost = {2}".format(
                    InstanceName.TRAINING, steps, used_time))
                logging.info("current loss: {0}".format(fetch_output_dict['loss']))
                time_begin = time.time()

            if self.model_class.lr_scheduler:
                self.model_class.lr_scheduler.step()

            if self.trainer_id == 0:
                if steps % self.params["save_model_step"] == 0:
                    self.save_model(steps)
            steps += 1

        if self.trainer_id == 0:
            self.save_model(steps)


    def do_visual(self):
        """评估指标的可视化展示
        """
        pass

