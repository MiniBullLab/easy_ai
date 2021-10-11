#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.helper import DirProcess
from easyai.helper.data_structure import OCRObject
from easyai.name_manager.evaluation_name import EvaluationName
from easyai.tools.offline_test.base_offline_evaluation import BaseOfflineEvaluation
from easyai.helper.arguments_parse import ToolArgumentsParse
from easyai.name_manager.task_name import TaskName
from easyai.tools.utility.tools_registry import REGISTERED_OFFLINE_EVALUATION


@REGISTERED_OFFLINE_EVALUATION.register_module(TaskName.RecognizeText)
class OfflineRecognizeTextEvaluation(BaseOfflineEvaluation):

    def __init__(self):
        super().__init__()
        self.dir_process = DirProcess()
        self.evaluation_args = {"type": EvaluationName.RecognizeTextMetric}
        self.evaluation = self.evaluation_factory.get_evaluation(self.evaluation_args)

    def process(self, test_path, target_path):
        self.evaluation.reset()
        test_data_list = self.get_data(test_path)
        target_data_list = self.get_data(target_path)
        for image_name, text_data in test_data_list:
            for target_name, target_text in target_data_list:
                if image_name in target_name:
                    self.evaluation.eval([text_data], [target_text])
                    break
        average_socre = self.evaluation.get_score()
        return average_socre['accuracy']

    def get_data(self, data_path):
        result = []
        for line_data in self.dir_process.getFileData(data_path):
            data_list = [x.strip() for x in line_data.split('|', 1) if x.strip()]
            if len(data_list) == 2:
                # print(data_list[0])
                ocr_object = OCRObject()
                ocr_object.set_text(data_list[1].strip())
                result.append((data_list[0], ocr_object))
        return result

    def print_evaluation(self, value):
        print_str = "Accuracy: {:.3f}%".format(value * 100)
        print(print_str)
        return print_str


def main():
    print("start...")
    options = ToolArgumentsParse.test_path_parse()
    test = OfflineRecognizeTextEvaluation()
    value = test.process(options.inputPath, options.targetPath)
    test.print_evaluation(value)
    print("End of game, have a nice day!")


if __name__ == "__main__":
   main()




