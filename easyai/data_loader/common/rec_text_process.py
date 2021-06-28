#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
from easyai.utility.logger import EasyLogger


class RecTextProcess():

    def __init__(self):
        self.text_dict = {}

    def read_character(self, char_path):
        if not os.path.exists(char_path):
            EasyLogger.error("char_path(%s) not exists" % char_path)
            return
        dict_character = []
        with open(char_path, "rb") as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.decode('utf-8').strip("\n").strip("\r\n")
                dict_character += list(line)
        self.text_dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'blank' token required by CTCLoss
            self.text_dict[char] = i + 1
        # TODO replace ‘ ’ with special symbol
        # dummy '[blank]' token for CTCLoss (index 0)
        character = ['[blank]'] + dict_character
        EasyLogger.debug("text_dict: {}".format(self.text_dict))
        EasyLogger.debug("character count: %d" % len(character))
        EasyLogger.debug(character)
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
