#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
from optparse import OptionParser


class TaskArgumentsParse():

    def __init__(self):
        pass

    @classmethod
    def multi_tracker_parse_arguments(cls):

        parser = OptionParser()
        parser.description = "This program task"

        parser.add_option("-r", "--reid", dest="reid_name",
                          type="string", default=None,
                          help="reid task name")

        parser.add_option("-t", "--tracker", dest="tracker_name",
                          type="string", default=None,
                          help="multi tracker name")

        parser.add_option("-i", "--input", dest="inputPath",
                          metavar="PATH", type="string", default=None,
                          help="images path or video path")

        parser.add_option("-w", "--weights", dest="weights", action="append",
                          metavar="PATH", default=[],
                          help="path to store weights")

        parser.add_option("-c", "--config", dest="config_path",
                          metavar="PATH", type="string", default=None,
                          help="config path")

        parser.add_option("-s", "--show", action="store_true",
                          dest="show",
                          default=False,
                          help="show result")

        parser.add_option("-d", "--data_type", dest="data_type",
                          action="store", type="int", default=1,
                          help="input data type(none, images, video)")

        (options, args) = parser.parse_args()

        if options.data_type != 0:
            if options.inputPath:
                if not os.path.exists(options.inputPath):
                    parser.error("Could not find the input file")
                else:
                    options.input_path = os.path.normpath(options.inputPath)
            else:
                parser.error("'input' option is required to run this program")
        return options
