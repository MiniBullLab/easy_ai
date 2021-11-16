#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: lipeijie

from easyai.utility.logger import EasyLogger
log_file_path = EasyLogger.get_log_file_path("ai_runtime.log")
EasyLogger.init(logfile_level="debug", log_file=log_file_path, stdout_level="error")

import os
import inspect
from optparse import OptionParser
from easy_tools.model_train.ai_train import EasyAiModelTrain


def parse_arguments():
    parser = OptionParser()
    parser.description = "This program train model"

    parser.add_option("-t", "--task", dest="task_name",
                      type="string", default=None,
                      help="task name")

    parser.add_option("-g", "--gpu", dest="gpu_id",
                      type="int", default=0,
                      help="gpu id")

    parser.add_option("-i", "--trainPath", dest="trainPath",
                      metavar="PATH", type="string", default="./train.txt",
                      help="path to data config file")

    parser.add_option("-v", "--valPath", dest="valPath",
                      metavar="PATH", type="string", default="./val.txt",
                      help="path to data config file")

    (options, args) = parser.parse_args()

    EasyLogger.debug(options)

    # if options.trainPath:
    #     if not os.path.exists(options.trainPath):
    #         parser.error("Could not find the input train file")
    #     else:
    #         options.input_path = os.path.normpath(options.trainPath)
    # else:
    #     parser.error("'trainPath' option is required to run this program")

    return options


def train_main():
    EasyLogger.info("easyai process start...")
    options = parse_arguments()
    current_path = inspect.getfile(inspect.currentframe())
    dir_name = os.path.dirname(current_path)
    train_process = EasyAiModelTrain(options.trainPath, options.valPath, options.gpu_id)

    if options.task_name.strip() == "NG_OK":
        train_process.binary_classidy_model_train(dir_name)
    elif options.task_name.strip() == "ClassNet":
        train_process.classify_model_train(dir_name)
    elif options.task_name.strip() == "DeNet":
        train_process.det2d_model_train(dir_name)
    elif options.task_name.strip() == "SegNet":
        train_process.segment_model_train(dir_name)
    elif options.task_name.strip() == "TextNet":
        train_process.rec_text_model_train(dir_name)
    elif options.task_name.strip() == "OneClass":
        train_process.one_class_model_train(dir_name)
    elif options.task_name.strip() == "OCRDenet":
        train_process.ocr_denet_model_train(dir_name)
    else:
        EasyLogger.error("input task error!")
    EasyLogger.info("easyai process end!")


if __name__ == "__main__":
    train_main()
