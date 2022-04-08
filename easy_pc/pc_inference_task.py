#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import traceback
from easyai.utility.logger import EasyLogger
if EasyLogger.check_init():
    log_file_path = EasyLogger.get_log_file_path("pc_inference.log")
    EasyLogger.init(logfile_level="debug", log_file=log_file_path, stdout_level="error")


from easyai.tasks.utility.task_registry import REGISTERED_INFERENCE_TASK
from easyai.utility.registry import build_from_cfg
from easyai.helper.arguments_parse import TaskArgumentsParse


class PCInferenceTask():

    def __init__(self, task_name, input_path, data_type, is_show=True):
        self.task_name = task_name
        self.input_path = input_path
        self.data_type = data_type
        self.is_show = is_show

    def infer(self, model_name, gpu_id, weight_path, config_path):
        task_args = {'type': self.task_name,
                     'model_name': model_name,
                     'gpu_id': gpu_id,
                     'config_path': config_path}
        EasyLogger.debug(task_args)
        EasyLogger.debug(weight_path)
        if self.task_name is not None and \
                REGISTERED_INFERENCE_TASK.has_class(self.task_name):
            try:
                task = build_from_cfg(task_args, REGISTERED_INFERENCE_TASK)
                task.load_weights(weight_path)
                task.process(self.input_path, self.data_type, self.is_show)
            except Exception as err:
                EasyLogger.error(traceback.format_exc())
                EasyLogger.error(err)
        else:
            EasyLogger.error("This task(%s) not exits!" % self.task_name)


def main():
    EasyLogger.info("Inference process start...")
    options = TaskArgumentsParse.inference_parse_arguments()
    inference_task = PCInferenceTask(options.task_name, options.inputPath,
                                     options.data_type, options.show)
    inference_task.infer(options.model, 0, options.weights, options.config_path)
    EasyLogger.info("Inference process end!")


if __name__ == '__main__':
    main()
