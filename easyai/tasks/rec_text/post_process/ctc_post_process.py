#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import numpy as np
from easyai.helper.data_structure import OCRObject
from easyai.tasks.utility.base_post_process import BasePostProcess
from easyai.name_manager.post_process_name import PostProcessName
from easyai.tasks.utility.task_registry import REGISTERED_POST_PROCESS


@REGISTERED_POST_PROCESS.register_module(PostProcessName.CTCPostProcess)
class CTCPostProcess(BasePostProcess):

    def __init__(self):
        super().__init__()

    def __call__(self, prediction, character):
        """ convert text-index into text-label. """
        if prediction.ndim == 2:
            prediction = np.expand_dims(prediction, 0)
        # print(prediction.shape)
        blank_index = 0
        preds_idx = prediction.argmax(axis=2)
        preds_prob = prediction.max(axis=2)
        result_list = []
        for word, prob in zip(preds_idx, preds_prob):
            result = []
            conf = []
            temp_object = OCRObject()
            for i, index in enumerate(word):
                if word[i] != blank_index and (not (i > 0 and word[i - 1] == word[i])):
                    result.append(character[int(index)])
                    conf.append(prob[i])
            temp_object.set_text(''.join(result))
            temp_object.text_confidence = conf
            result_list.append(temp_object)
        return result_list
