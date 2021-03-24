#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.tasks.utility.registry import REGISTERED_INFERENCE_TASK
from easyai.utility.registry import build_from_cfg
from easyai.helper.arguments_parse import TaskArgumentsParse


class InferenceTask():

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
        # print(model_name, weight_path)
        if self.task_name is not None and \
                REGISTERED_INFERENCE_TASK.has_class(self.task_name):
            task = build_from_cfg(task_args, REGISTERED_INFERENCE_TASK)
            task.load_weights(weight_path)
            task.process(self.input_path, self.data_type, self.is_show)
        else:
            print("This task(%s) not exits!" % self.task_name)


def main():
    print("process start...")
    options = TaskArgumentsParse.inference_parse_arguments()
    inference_task = InferenceTask(options.task_name, options.inputPath,
                                   options.data_type, options.show)
    inference_task.infer(options.model, 0, options.weights, options.config_path)
    print("process end!")


if __name__ == '__main__':
    main()
