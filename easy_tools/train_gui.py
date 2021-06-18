#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

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