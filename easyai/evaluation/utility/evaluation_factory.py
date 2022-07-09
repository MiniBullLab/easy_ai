#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import traceback
from easyai.utility.logger import EasyLogger
from easyai.evaluation.utility.evaluation_registry import REGISTERED_EVALUATION
from easyai.utility.registry import build_from_cfg


class EvaluationFactory():

    def __init__(self):
        pass

    def get_evaluation(self, evaluation_args):
        result = None
        EasyLogger.debug(evaluation_args)
        try:
            type_name = evaluation_args['type'].strip()
            if REGISTERED_EVALUATION.has_class(type_name):
                result = build_from_cfg(evaluation_args, REGISTERED_EVALUATION)
            else:
                EasyLogger.error("%s evaluation not exits" % type_name)
        except ValueError as err:
            EasyLogger.error(traceback.format_exc())
            EasyLogger.error(err)
        except TypeError as err:
            EasyLogger.error(traceback.format_exc())
            EasyLogger.error(err)
        return result
