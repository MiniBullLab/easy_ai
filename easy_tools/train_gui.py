#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.utility.logger import EasyLogger
log_file_path = EasyLogger.get_log_file_path("train_gui.log")
EasyLogger.init(logfile_level="debug", log_file=log_file_path, stdout_level="error")

import sys
from PyQt5.QtWidgets import QApplication
from easy_tools.model_train.ai_train_window import EasyAiTrainWindow


def main():
    app = QApplication(sys.argv)
    window = EasyAiTrainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()