#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import logging
from logging.handlers import TimedRotatingFileHandler

__all__ = ['EasyLogger']

DEFAULT_LOGFILE_LEVEL = 'debug'
DEFAULT_STDOUT_LEVEL = 'info'
DEFAULT_LOG_FILE = './default.log'
DEFAULT_LOG_FORMAT = '%(asctime)s %(levelname)-7s %(message)s'

LOG_LEVEL_DICT = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL
}


class EasyLogger():
    """
    Args:
      Log level: CRITICAL>ERROR>WARNING>INFO>DEBUG.
      Log file: The file that stores the logging info.
      rewrite: Clear the log file.
      log format: The format of log messages.
      stdout level: The log level to print on the screen.
    """
    ROOT_DIR = "./.easy_log"

    logfile_level = None
    log_file = None
    log_format = None
    rewrite = None
    stdout_level = None
    logger = None

    @staticmethod
    def init(logfile_level=DEFAULT_LOGFILE_LEVEL,
             log_file=DEFAULT_LOG_FILE,
             log_format=DEFAULT_LOG_FORMAT,
             rewrite=False,
             stdout_level=DEFAULT_STDOUT_LEVEL):
        EasyLogger.logfile_level = logfile_level
        EasyLogger.log_file = log_file
        EasyLogger.log_format = log_format
        EasyLogger.rewrite = rewrite
        EasyLogger.stdout_level = stdout_level

        EasyLogger.logger = logging.getLogger()
        fmt = logging.Formatter(EasyLogger.log_format)

        if EasyLogger.logfile_level is not None:
            filemode = 'w'
            if not EasyLogger.rewrite:
                filemode = 'a'

            dir_name = os.path.dirname(os.path.abspath(EasyLogger.log_file))
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

            if EasyLogger.logfile_level not in LOG_LEVEL_DICT:
                print('Invalid logging level: {}'.format(EasyLogger.logfile_level))
                EasyLogger.logfile_level = DEFAULT_LOGFILE_LEVEL

            EasyLogger.logger.setLevel(LOG_LEVEL_DICT[EasyLogger.logfile_level])

            # file_handler = logging.FileHandler(EasyLogger.log_file, mode=filemode)
            # file_handler.setFormatter(fmt)
            # file_handler.setLevel(LOG_LEVEL_DICT[EasyLogger.logfile_level])
            # EasyLogger.logger.addHandler(file_handler)

            file_handler = logging.handlers.RotatingFileHandler(
                filename=EasyLogger.log_file,
                maxBytes=1024 * 1024 * 50,
                backupCount=3
            )
            file_handler.setFormatter(fmt)
            file_handler.setLevel(LOG_LEVEL_DICT[EasyLogger.logfile_level])
            EasyLogger.logger.addHandler(file_handler)
            EasyLogger.logger.addHandler(file_handler)

            # time_handler = TimedRotatingFileHandler(filename=EasyLogger.log_file,
            #                                         when="D",
            #                                         interval=1,
            #                                         backupCount=3)
            # EasyLogger.logger.addHandler(time_handler)

        if stdout_level is not None:
            if EasyLogger.logfile_level is None:
                EasyLogger.logger.setLevel(LOG_LEVEL_DICT[EasyLogger.stdout_level])

            console = logging.StreamHandler()
            if EasyLogger.stdout_level not in LOG_LEVEL_DICT:
                print('Invalid logging level: {}'.format(EasyLogger.stdout_level))
                return

            console.setLevel(LOG_LEVEL_DICT[EasyLogger.stdout_level])
            console.setFormatter(fmt)
            EasyLogger.logger.addHandler(console)

    @staticmethod
    def get_log_file_path(file_name):
        return os.path.join(EasyLogger.ROOT_DIR, file_name)

    @staticmethod
    def set_log_file(file_path):
        EasyLogger.log_file = file_path
        EasyLogger.init(log_file=file_path)

    @staticmethod
    def set_logfile_level(log_level):
        if log_level not in LOG_LEVEL_DICT:
            print('Invalid logging level: {}'.format(log_level))
            return
        EasyLogger.init(logfile_level=log_level)

    @staticmethod
    def clear_log_file():
        EasyLogger.rewrite = True
        EasyLogger.init(rewrite=True)

    @staticmethod
    def check_logger():
        if EasyLogger.logger is None:
            EasyLogger.init(logfile_level=None, stdout_level=DEFAULT_STDOUT_LEVEL)

    @staticmethod
    def set_stdout_level(log_level):
        if log_level not in LOG_LEVEL_DICT:
            print('Invalid logging level: {}'.format(log_level))
            return
        EasyLogger.init(stdout_level=log_level)

    @staticmethod
    def debug(message):
        EasyLogger.check_logger()
        filename = os.path.basename(sys._getframe().f_back.f_code.co_filename)
        lineno = sys._getframe().f_back.f_lineno
        prefix = '[{}, {}]'.format(filename, lineno)
        EasyLogger.logger.debug('{} {}'.format(prefix, message))

    @staticmethod
    def info(message):
        EasyLogger.check_logger()
        filename = os.path.basename(sys._getframe().f_back.f_code.co_filename)
        lineno = sys._getframe().f_back.f_lineno
        prefix = '[{}, {}]'.format(filename, lineno)
        EasyLogger.logger.info('{} {}'.format(prefix, message))

    @staticmethod
    def warn(message):
        EasyLogger.check_logger()
        filename = os.path.basename(sys._getframe().f_back.f_code.co_filename)
        lineno = sys._getframe().f_back.f_lineno
        prefix = '[{}, {}]'.format(filename, lineno)
        EasyLogger.logger.warn('{} {}'.format(prefix, message))

    @staticmethod
    def error(message):
        EasyLogger.check_logger()
        filename = os.path.basename(sys._getframe().f_back.f_code.co_filename)
        lineno = sys._getframe().f_back.f_lineno
        prefix = '[{}, {}]'.format(filename, lineno)
        EasyLogger.logger.error('{} {}'.format(prefix, message))

    @staticmethod
    def critical(message):
        EasyLogger.check_logger()
        filename = os.path.basename(sys._getframe().f_back.f_code.co_filename)
        lineno = sys._getframe().f_back.f_lineno
        prefix = '[{}, {}]'.format(filename, lineno)
        EasyLogger.logger.critical('{} {}'.format(prefix, message))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logfile_level', default="debug", type=str,
                        dest='logfile_level', help='To set the log level to files.')
    parser.add_argument('--stdout_level', default="info", type=str,
                        dest='stdout_level', help='To set the level to print to screen.')
    parser.add_argument('--log_file', default="./default.log", type=str,
                        dest='log_file', help='The path of log files.')
    parser.add_argument('--log_format', default="%(asctime)s %(levelname)-7s %(message)s",
                        type=str, dest='log_format', help='The format of log messages.')
    parser.add_argument('--rewrite', default=False, type=bool,
                        dest='rewrite', help='Clear the log files existed.')

    args = parser.parse_args()
    EasyLogger.init(logfile_level=args.logfile_level, stdout_level=args.stdout_level,
                    log_file=args.log_file, log_format=args.log_format, rewrite=args.rewrite)

    EasyLogger.info("info test.")
    EasyLogger.debug("debug test.")
    EasyLogger.warn("warn test.")
    EasyLogger.error("error test.")
    EasyLogger.debug("debug test.")
    EasyLogger.debug({"a": [1, 2, 3]})


if __name__ == "__main__":
    main()