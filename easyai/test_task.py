#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.utility.logger import EasyLogger
log_file_path = EasyLogger.get_log_file_path("test.log")
EasyLogger.init(logfile_level="debug", log_file=log_file_path, stdout_level="error")


from easyai.tasks.utility.task_registry import REGISTERED_TEST_TASK
from easyai.utility.registry import build_from_cfg
from easyai.helper.arguments_parse import TaskArgumentsParse


class TestTask():

    def __init__(self, task_name, val_path):
        self.val_path = val_path
        self.task_name = task_name

    def test(self, model_name, gpu_id, weight_path, config_path):
        task_args = {'type': self.task_name,
                     'model_name': model_name,
                     'gpu_id': gpu_id,
                     'config_path': config_path}
        EasyLogger.debug(task_args)
        EasyLogger.debug(weight_path)
        if self.task_name is not None and \
                REGISTERED_TEST_TASK.has_class(self.task_name):
            # try:
            task = build_from_cfg(task_args, REGISTERED_TEST_TASK)
            task.load_weights(weight_path)
            task.test(self.val_path)
            # except Exception as err:
            #     EasyLogger.error(err)
        else:
            EasyLogger.error("This task(%s) not exits!" % self.task_name)


def main():
    EasyLogger.info("Test process start...")
    options = TaskArgumentsParse.test_input_parse()
    test_task = TestTask(options.task_name, options.valPath)
    test_task.test(options.model, 0, options.weights, options.config_path)
    EasyLogger.info("Test process end!")


if __name__ == '__main__':
    main()
