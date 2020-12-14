#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: lipeijie

import os
import inspect
from optparse import OptionParser
from easy_tools.easyai_train import EasyAiModelTrain


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

    parser.add_option("-c", "--config", dest="config_path",
                      metavar="PATH", type="string", default=None,
                      help="config path")

    (options, args) = parser.parse_args()

    # if options.trainPath:
    #     if not os.path.exists(options.trainPath):
    #         parser.error("Could not find the input train file")
    #     else:
    #         options.input_path = os.path.normpath(options.trainPath)
    # else:
    #     parser.error("'trainPath' option is required to run this program")

    return options


def train_main():
    print("process start...")
    options = parse_arguments()
    current_path = inspect.getfile(inspect.currentframe())
    dir_name = os.path.dirname(current_path)
    train_process = EasyAiModelTrain(options.trainPath, options.valPath, options.gpu_id, options.config_path)

    if options.task_name.strip() == "ClassNet":
        train_process.classify_model_train(dir_name)
    elif options.task_name.strip() == "DeNET":
        train_process.det2d_model_train(dir_name)
    elif options.task_name.strip() == "SegNET":
        train_process.segment_model_train(dir_name)
    else:
        print("input task error!")
    print("process end!")


if __name__ == "__main__":
    train_main()
