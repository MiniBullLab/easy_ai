#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import cv2
from easyai.utility.logger import EasyLogger
if EasyLogger.check_init():
    log_file_path = EasyLogger.get_log_file_path("tools.log")
    EasyLogger.init(logfile_level="debug", log_file=log_file_path, stdout_level="error")
from easyai.tasks.utility.task_registry import REGISTERED_INFERENCE_TASK
from easyai.utility.registry import build_from_cfg
from easyai.helper.arguments_parse import TaskArgumentsParse


class BotInference():

    def __init__(self, task_name, data_type):
        self.task_name = task_name
        self.data_type = data_type
        self.task = None

    def build_task(self, model_name, gpu_id, weight_path, config_path):
        task_args = {'type': self.task_name,
                     'model_name': model_name,
                     'gpu_id': gpu_id,
                     'config_path': config_path}
        EasyLogger.debug(task_args)
        EasyLogger.debug(weight_path)
        if self.task_name is not None and \
                REGISTERED_INFERENCE_TASK.has_class(self.task_name):
            try:
                self.task = build_from_cfg(task_args, REGISTERED_INFERENCE_TASK)
                self.task.load_weights(weight_path)
            except Exception as err:
                EasyLogger.error(err)
        else:
            EasyLogger.info("This task(%s) not exits!" % self.task_name)

    def infer(self, numpy_data):
        result = None
        try:
            input_data = self.task.get_single_image_data(numpy_data)
            result = self.task.single_image_process(input_data)
        except Exception as err:
            EasyLogger.error(err)
        return result


def main():
    EasyLogger.info("Inference process start...")
    options = TaskArgumentsParse.inference_parse_arguments()
    src_image = cv2.imread(options.inputPath)  # BGR
    inference_task = BotInference(options.task_name, options.data_type)
    inference_task.build_task(options.model, 0, options.weights, options.config_path)
    inference_task.infer(src_image)
    EasyLogger.info("Inference process end!")


if __name__ == '__main__':
    main()
