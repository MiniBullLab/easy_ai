#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import torch
import numpy as np
from easyai.data_loader.utility.image_dataset_process import ImageDataSetProcess


class SegmentResultProcess():

    def __init__(self):
        self.dataset_process = ImageDataSetProcess()

    def get_segmentation_result(self, prediction, threshold=0):
        result = None
        if prediction.ndim == 2:
            result = (prediction >= threshold).astype(int)
            # print(set(list(result.flatten())))
        elif prediction.ndim == 3:
            result = np.argmax(prediction, axis=0)
        elif prediction.ndim == 4:
            result = np.argmax(prediction, axis=1)
        return result

    def resize_segmention_result(self, src_size, image_size,
                                 resize_type, segmention_result):
        if src_size[0] == 0 or src_size[1] == 0:
            return None
        else:
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
