#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.helper.arguments_parse import TaskArgumentsParse
from easyai.tasks.cls.classify_test import ClassifyTest
from easyai.tasks.det2d.detect2d_test import Detection2dTest
from easyai.tasks.seg.segment_test import SegmentionTest
from easyai.tasks.multi_task.det2d_seg_task_test import Det2dSegTaskTest
from easyai.base_name.task_name import TaskName


class TestTask():

    def __init__(self, val_path, weight_path):
        self.val_path = val_path
        self.weight_path = weight_path

    def segment_task(self, cfg_path, gpu_id, config_path):
        seg_test = SegmentionTest(cfg_path, gpu_id, config_path)
        seg_test.load_weights(self.weight_path)
        seg_test.test(self.val_path)


def main():
    print("process start...")
    options = TaskArgumentsParse.test_input_parse()
    test_task = TestTask(options.valPath, options.weights)
    if options.task_name == TaskName.Classify_Task:
        test_task.classify_task(options.model, 0, options.config_path)
    elif options.task_name == TaskName.Detect2d_Task:
        test_task.detect2d_task(options.model, 0, options.config_path)
    elif options.task_name == TaskName.Segment_Task:
        test_task.segment_task(options.model, 0, options.config_path)
    elif options.task_name == TaskName.Det2d_Seg_Task:
        test_task.det2d_seg_task(options.model, 0, options.config_path)
    print("process end!")


if __name__ == '__main__':
    main()
