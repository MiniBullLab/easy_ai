#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import numpy as np
from easyai.name_manager.post_process_name import PostProcessName
from easyai.tasks.utility.task_registry import REGISTERED_POST_PROCESS


@REGISTERED_POST_PROCESS.register_module(PostProcessName.CTCPostProcess)
class CTCPostProcess():

    def __init__(self):
        pass

    def __call__(self, prediction, character):
        """ convert text-index into text-label. """
        prediction = np.expand_dims(prediction, 0)
        preds_idx = prediction.argmax(axis=2)
        preds_prob = prediction.max(axis=2)
        result_list = []
        for word, prob in zip(preds_idx, preds_prob):
            result = []
            conf = []
            for i, index in enumerate(word):
                if word[i] != 0 and (not (i > 0 and word[i - 1] == word[i])):
                    result.append(self.character[int(index)])
                    conf.append(prob[i])
            result_list.append((''.join(result), conf))
        return result_list
