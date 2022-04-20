#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie


import os
import numpy as np
import traceback
from easyai.utility.logger import EasyLogger
if EasyLogger.check_init():
    log_file_path = EasyLogger.get_log_file_path("tools.log")
    EasyLogger.init(logfile_level="debug", log_file=log_file_path, stdout_level="error")
from easyai.tasks.utility.base_test import BaseTest
from easyai.name_manager.task_name import TaskName
from easyai.tasks.one_class.one_class import OneClass
from easyai.tasks.one_class.one_class_result_process import OneClassResultProcess
from easyai.utility.logger import EasyLogger
from easyai.helper.arguments_parse import TaskArgumentsParse


class TempFeatureSave(BaseTest):

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
        for bin_name in os.listdir(".easy_log/bin_all/"):
            prediction = np.fromfile(os.path.join(".easy_log/bin_all/", bin_name), dtype=np.float32)
            square_dim = np.sqrt((len(prediction)) / 1440)
            prediction = prediction.reshape((1440, int(square_dim), int(square_dim)))
            if prediction.ndim == 3:
                prediction = np.expand_dims(prediction, axis=0)
            self.process_func.add_embedding(prediction)
        self.process_func.save_embedding()


def main():
    EasyLogger.info("Test process start...")
    try:
        options = TaskArgumentsParse.test_input_parse()
        task = TempFeatureSave(options.model, 0, options.config_path)
        task.load_weights(options.weights)
        task.process_test(options.valPath)
    except Exception as err:
        EasyLogger.error(traceback.format_exc())
        EasyLogger.error(err)
    EasyLogger.info("Test process end!")


if __name__ == '__main__':
    main()