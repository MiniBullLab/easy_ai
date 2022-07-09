#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
from easyai.helper.image_process import ImageProcess
from easyai.tools.sample_tool.convert_segment_label import ConvertSegmentionLable
from easyai.data_loader.seg.segment_sample import SegmentSample
from easyai.helper.arguments_parse import ToolArgumentsParse
from easyai.tools.eval_tool.offline_test.base_offline_evaluation import BaseOfflineEvaluation
from easyai.name_manager.evaluation_name import EvaluationName
from easyai.name_manager.task_name import TaskName
from easyai.tools.utility.tools_registry import REGISTERED_OFFLINE_EVALUATION


@REGISTERED_OFFLINE_EVALUATION.register_module(TaskName.Segment_Task)
class OfflineSegmentEvaluation(BaseOfflineEvaluation):

    def __init__(self, seg_label_type=0, segment_class=None):
        super().__init__()
        self.seg_label_type = seg_label_type
        self.segment_class = segment_class
        self.image_process = ImageProcess()
        self.label_converter = ConvertSegmentionLable()
        self.segment_sample = SegmentSample(None)
        self.evaluation_args = {"type": EvaluationName.SegmentionMetric,
                                'num_class': len(self.segment_class)}
        self.evaluation = self.evaluation_factory.get_evaluation(self.evaluation_args)

    def process(self, test_path, target_path):
        self.evaluation.reset()
        test_dir = test_path
        target_data_list = self.segment_sample.get_image_and_label_list(target_path)
        for image_path, label_path in target_data_list:
            path, filename_post = os.path.split(label_path)
            test_path = os.path.join(test_dir, filename_post)
            test_data = self.read_label_image(test_path, 0)
            target_data = self.read_label_image(label_path, self.seg_label_type)
            self.evaluation.numpy_eval(test_data, target_data)
        score, class_score = self.evaluation.get_score()
        return score

    def read_label_image(self, label_path, label_type):
        if label_type == 0:
            mask = self.image_process.read_gray_image(label_path)
        else:
            mask = self.label_converter.process_segment_label(label_path,
                                                              label_type,
                                                              self.segment_class)
        return mask

    def print_evaluation(self, value):
        for k, v in value.items():
            print(k, v)


def main():
    print("start...")
    from easyai.config.utility.config_factory import ConfigFactory
    options = ToolArgumentsParse.test_path_parse()
    config_factory = ConfigFactory()
    task_config = config_factory.get_config(TaskName.Segment_Task, config_path=options.config_path)
    test = OfflineSegmentEvaluation(task_config.seg_label_type,
                                    task_config.segment_class)
    value = test.process(options.inputPath, options.targetPath)
    test.print_evaluation(value)
    print("End of game, have a nice day!")


if __name__ == "__main__":
   main()


