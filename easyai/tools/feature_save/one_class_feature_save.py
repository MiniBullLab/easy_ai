#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.tasks.utility.base_test import BaseTest
from easyai.name_manager.task_name import TaskName
from easyai.tasks.one_class.one_class import OneClass
from easyai.tasks.one_class.one_class_result_process import OneClassResultProcess
from easyai.utility.logger import EasyLogger
from easyai.helper.arguments_parse import TaskArgumentsParse


class OneClassFeatureSave(BaseTest):

    def __init__(self, model_name, gpu_id, config_path=None):
        super().__init__(TaskName.OneClass)
        self.inference = OneClass(model_name, gpu_id, config_path)
        self.set_test_config(self.inference.task_config)
        self.set_model()
        result_process = OneClassResultProcess(self.test_task_config.post_process)
        self.process_func = result_process.process_func

    def process_test(self, data_path, epoch=0):
        self.create_dataloader(data_path)
        if not self.start_test():
            EasyLogger.info("no test!")
            return
        self.test(epoch)

    def test(self, epoch=0):
        self.process_func.reset()
        for index, batch_data in enumerate(self.dataloader):
            prediction, _ = self.inference.infer(batch_data)
            self.process_func.add_embedding(prediction)
        self.process_func.save_embedding()

    def create_dataloader(self, data_path):
        assert self.test_task_config is not None
        dataloader_config = self.test_task_config.train_data.get('dataloader', None)
        dataset_config = self.test_task_config.train_data.get('dataset', None)
        self.dataloader = self.dataloader_factory.get_train_dataloader(data_path,
                                                                       dataloader_config,
                                                                       dataset_config)
        self.batch_data_process_func = \
            self.batch_data_process_factory.build_process(self.test_task_config.batch_data_process)
        if self.dataloader is not None:
            self.total_batch_data = len(self.dataloader)
        else:
            self.total_batch_data = 0


def main():
    EasyLogger.info("Test process start...")
    options = TaskArgumentsParse.test_input_parse()
    task = OneClassFeatureSave(options.model, 0, options.config_path)
    task.load_weights(options.weights)
    task.process_test(options.valPath)
    EasyLogger.info("Test process end!")


if __name__ == '__main__':
    main()
