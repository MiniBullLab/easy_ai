#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
import collections
import string
from easyai.utility.logger import EasyLogger


class RecTextProcess():

    def __init__(self, use_space=False):
        self.use_space = use_space
        self.text_dict = collections.OrderedDict()
        # for i, char in enumerate(string.printable[:-6]):
        #     # NOTE: 0 is reserved for 'blank' token required by CTCLoss
        #     self.text_dict[char] = i + 1

    def read_character(self, char_path):
        if not os.path.exists(char_path):
            EasyLogger.error("char_path(%s) not exists" % char_path)
            return []
        character_list = []
        with open(char_path, "rb") as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.decode('utf-8').strip("\n").strip("\r\n")
                character_list += list(line)
        if self.use_space:
            character_list += [' ']
        self.text_dict = {}
        for i, char in enumerate(character_list):
            # NOTE: 0 is reserved for 'blank' token required by CTCLoss
            self.text_dict[char] = i + 1
        # '[blank]' token for CTCLoss (index 0)
        character_set = self.text_dict.keys()
        character = ['[blank]'] + list(character_set)
        EasyLogger.debug("character count: %d" % len(character))
        EasyLogger.debug(character)
        EasyLogger.debug("text_dict: {}".format(self.text_dict))
        return character

    def text_encode(self, text):
        final_text = []
        text_code = []
        for char in text:
            index = self.text_dict.get(char, 0)
            if index != 0:
                final_text.append(char)
                text_code.append(index)
            else:
                EasyLogger.warn("Error char: %s(%s)" % (text, char))
        return text_code, ''.join(final_text)
