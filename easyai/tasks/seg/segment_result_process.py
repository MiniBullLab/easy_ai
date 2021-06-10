#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import torch
import numpy as np
from easyai.tasks.utility.task_result_process import TaskPostProcess


class SegmentResultProcess(TaskPostProcess):

    def __init__(self, image_size, resize_type, post_process_args):
        super().__init__()
        self.image_size = image_size
        self.resize_type = resize_type
        self.process_func = self.build_post_process(post_process_args)

    def post_process(self, prediction, src_size=(0, 0)):
        if prediction is None:
            return None, None
        result = self.process_func(prediction)
        if src_size[0] == 0 or src_size[1] == 0:
            seg_image = None
        else:
            seg_image = self.resize_segmention_result(src_size,
                                                      self.image_size,
                                                      self.resize_type,
                                                      result)
        return result, seg_image

    def resize_segmention_result(self, src_size, image_size,
                                 resize_type, segmention_result):
        result = self.dataset_process.inv_resize(src_size, image_size,
                                                 resize_type, segmention_result)
        result = result.astype(np.float32)
        return result

    def output_feature_map_resize(self, input_data, target):
        n, c, h, w = input_data.size()
        nt, ht, wt = target.size()
        # Handle inconsistent size between input and target
        if h > ht and w > wt:  # upsample labels
            target = target.type(input_data.dtype)
            target = target.unsqueeze(1)
            target = torch.nn.functional.upsample(target, size=(h, w), mode='nearest')
            target = target.squeeze(1).long()
        elif h < ht and w < wt:  # upsample images
            input_data = torch.nn.functional.upsample(input_data, size=(ht, wt), mode='bilinear')
        elif h == ht and w == wt:
            pass
        else:
            print("input_data: (%d,%d) and target: (%d,%d) error "
                  % (h, w, ht, wt))
            raise Exception("segment_data_resize error")
        return input_data, target
