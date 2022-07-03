#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import traceback
from easyai.utility.logger import EasyLogger
if EasyLogger.check_init():
    log_file_path = EasyLogger.get_log_file_path("pc_train.log")
    EasyLogger.init(logfile_level="debug", log_file=log_file_path, stdout_level="error")

from easyai.tasks.utility.task_registry import REGISTERED_TRAIN_TASK
from easyai.utility.registry import build_from_cfg
from easyai.helper.arguments_parse import TaskArgumentsParse


class PCTrainTask():

    def __init__(self, task_name, train_path, val_path):
        self.task_name = task_name
        self.train_path = train_path
        self.val_path = val_path

        self.convert_input_names = None
        self.convert_output_names = None
        self.is_convert = True
        self.save_onnx_path = None

    def train(self, model_name, gpu_id, config_path, pretrain_model_path):
        task_args = {'type': self.task_name,
                     'model_name': model_name,
                     'gpu_id': gpu_id,
                     'config_path': config_path}
        EasyLogger.debug(task_args)
        if self.task_name is not None and \
                REGISTERED_TRAIN_TASK.has_class(self.task_name):
            try:
                task = build_from_cfg(task_args, REGISTERED_TRAIN_TASK)
                task.load_pretrain_model(pretrain_model_path)
                task.train(self.train_path, self.val_path)
                self.image_model_convert(task, task.model_args)
            except Exception as err:
                EasyLogger.error(traceback.format_exc())
                EasyLogger.error(err)
        else:
            EasyLogger.error("This task(%s) not exits!" % self.task_name)

    def set_convert_param(self, is_convert, input_names, output_names):
        self.is_convert = is_convert
        self.convert_input_names = input_names
        self.convert_output_names = output_names

    def image_model_convert(self, train_task, model_args):
        if self.is_convert:
            from easyai.tools.model_tool.model_to_onnx import ModelConverter
            EasyLogger.debug(train_task.train_task_config.data)
            converter = ModelConverter(train_task.train_task_config.data['image_size'])
            self.save_onnx_path = converter.convert_process(model_args,
                                                            train_task.train_task_config.best_weights_path,
                                                            train_task.train_task_config.snapshot_dir,
                                                            self.convert_input_names,
                                                            self.convert_output_names)


def main():
    EasyLogger.info("process start...")
    options = TaskArgumentsParse.train_input_parse()
    train_task = PCTrainTask(options.task_name, options.trainPath, options.valPath)
    train_task.train(options.model, 0, options.config_path, options.pretrainModel)
    EasyLogger.info("process end!")


if __name__ == '__main__':
    main()
