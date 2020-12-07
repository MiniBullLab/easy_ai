#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.data_loader.utility.task_dataset_process import TaskDataSetProcess


class ClassifyDatasetProcess(TaskDataSetProcess):

    def __init__(self, resize_type, normalize_type,
                 mean=0, std=1, pad_color=0):
        super().__init__(resize_type, normalize_type, mean, std, pad_color)

