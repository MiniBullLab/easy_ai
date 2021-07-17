#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import numpy as np
from easyai.name_manager.task_name import TaskName
from easyai.config.utility.config_factory import ConfigFactory
from easyai.visualization.utility.task_show_factory import TaskShowFactory
from easyai.tasks.rec_text.text_result_process import TextResultProcess
from easyai.helper.arguments_parse import ToolArgumentsParse


class RecognizeTextARMInfer():

    def __init__(self, config_path=None):
        self.task_name = TaskName.RecognizeText
        self.config_path = config_path
        self.config_factory = ConfigFactory()
        self.task_config = self.config_factory.get_config(self.task_name, self.config_path)
        self.task_config.save_config()
        self.result_process = TextResultProcess(self.task_config.character_set,
                                                self.task_config.post_process)

    def process(self, feature_path):
        feature_map = self.read_feature_map(feature_path)
        text_objects = self.result_process.post_process(feature_map)
        print("result:", text_objects[0].get_text())

    def read_feature_map(self, feature_path):
        feature_map = np.fromfile(feature_path, dtype=np.float32)
        result = feature_map.reshape(self.task_config.train_data['dataset']['max_text_length'],
                                     self.task_config.character_count)
        return result


def main():
    print("start...")
    options = ToolArgumentsParse.infer_path_parse()
    test = RecognizeTextARMInfer(options.config_path)
    test.process(options.inputPath)
    print("End of game, have a nice day!")


if __name__ == "__main__":
   main()



