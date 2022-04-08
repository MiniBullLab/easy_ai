#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import traceback
from easyai.utility.logger import EasyLogger
if EasyLogger.check_init():
    log_file_path = EasyLogger.get_log_file_path("pc_test.log")
    EasyLogger.init(logfile_level="debug", log_file=log_file_path, stdout_level="error")


from easyai.tasks.utility.task_registry import REGISTERED_TEST_TASK
from easyai.utility.registry import build_from_cfg
from easyai.helper.arguments_parse import TaskArgumentsParse


class PCTestTask():

    def __init__(self, task_name, val_path):
        self.val_path = val_path
        self.task_name = task_name
        self.is_convert = False
        self.convert_input_names = None
        self.convert_output_names = None
        self.save_onnx_path = None

    def test(self, model_name, gpu_id, weight_path, config_path):
        task_args = {'type': self.task_name,
                     'model_name': model_name,
                     'gpu_id': gpu_id,
                     'config_path': config_path}
        EasyLogger.debug(task_args)
        EasyLogger.debug(weight_path)
        if self.task_name is not None and \
                REGISTERED_TEST_TASK.has_class(self.task_name):
            try:
                task = build_from_cfg(task_args, REGISTERED_TEST_TASK)
                task.load_weights(weight_path)
                task.process_test(self.val_path)
                self.image_model_convert(task, task.inference.model_args, weight_path)
            except Exception as err:
                EasyLogger.error(traceback.format_exc())
                EasyLogger.error(err)
        else:
            EasyLogger.error("This task(%s) not exits!" % self.task_name)

    def set_convert_param(self, is_convert, input_names, output_names):
        self.is_convert = is_convert
        self.convert_input_names = input_names
        self.convert_output_names = output_names

    def image_model_convert(self, test_task, model_args, weight_path):
        if self.is_convert:
            from easyai.tools.model_tool.model_to_onnx import ModelConverter
            EasyLogger.debug(test_task.test_task_config.data)
            converter = ModelConverter(test_task.test_task_config.data['image_size'])
            self.save_onnx_path = converter.convert_process(model_args,
                                                            weight_path,
                                                            test_task.test_task_config.snapshot_dir,
                                                            self.convert_input_names,
                                                            self.convert_output_names)


def main():
    EasyLogger.info("Test process start...")
    options = TaskArgumentsParse.test_input_parse()
    test_task = PCTestTask(options.task_name, options.valPath)
    test_task.test(options.model, 0, options.weights, options.config_path)
    EasyLogger.info("Test process end!")


if __name__ == '__main__':
    main()
