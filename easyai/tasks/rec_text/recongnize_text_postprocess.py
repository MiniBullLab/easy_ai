#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.tasks.utility.base_post_process import BasePostProcess
from easyai.tasks.utility.task_registry import REGISTERED_POST_PROCESS
from easyai.utility.registry import build_from_cfg


class Polygon2dPostProcess(BasePostProcess):

    def __init__(self, character_path, post_process_args):
        super().__init__()
        self.character_path = character_path
        self.post_process_args = post_process_args
        self.character = self.read_character(character_path)
        self.process_func = self.build_post_process(post_process_args)

    def post_process(self, prediction):
        if prediction is None:
            return None
        result = self.process_func(prediction, self.character)
        return result

    def build_post_process(self, post_process_args):
        func_name = post_process_args.strip()
        result_func = None
        if REGISTERED_POST_PROCESS.has_class(func_name):
            result_func = build_from_cfg(post_process_args, REGISTERED_POST_PROCESS)
        else:
            print("%s post process not exits" % func_name)
        return result_func

    def read_character(self, char_path):
        dict_character = []
        with open(char_path, "rb") as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.decode('utf-8').strip("\n").strip("\r\n")
                dict_character += list(line)
        # self.dict = {}
        # for i, char in enumerate(dict_character):
        #     # NOTE: 0 is reserved for 'blank' token required by CTCLoss
        #     self.dict[char] = i + 1
        # TODO replace ‘ ’ with special symbol
        # dummy '[blank]' token for CTCLoss (index 0)
        character = ['[blank]'] + dict_character + [' ']
        return character

