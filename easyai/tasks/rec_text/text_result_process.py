#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.utility.logger import EasyLogger
from easyai.tasks.utility.task_result_process import TaskPostProcess
from easyai.data_loader.common.rec_text_process import RecTextProcess
from easyai.tasks.utility.task_registry import REGISTERED_POST_PROCESS
from easyai.utility.registry import build_from_cfg


class TextResultProcess(TaskPostProcess):

    def __init__(self, character_path, post_process_args):
        super().__init__()
        self.character_path = character_path
        self.post_process_args = post_process_args
        self.text_process = RecTextProcess()
        self.character = self.text_process.read_character(character_path)
        self.process_func = self.build_post_process(post_process_args)
        EasyLogger.debug(character_path)

    def post_process(self, prediction):
        if prediction is None:
            return None
        result = self.process_func(prediction, self.character)
        return result

    def build_post_process(self, post_process_args):
        func_name = post_process_args['type'].strip()
        result_func = None
        if REGISTERED_POST_PROCESS.has_class(func_name):
            result_func = build_from_cfg(post_process_args, REGISTERED_POST_PROCESS)
        else:
            print("%s post process not exits" % func_name)
        return result_func
