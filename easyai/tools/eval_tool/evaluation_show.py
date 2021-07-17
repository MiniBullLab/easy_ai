#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import json
import codecs
import matplotlib.pyplot as plt
from easyai.evaluation.utility.classify_pr import ClassifyPrecisionAndRecall
from easyai.name_manager.task_name import TaskName
from easyai.config.utility.config_factory import ConfigFactory
from easyai.helper.arguments_parse import ToolArgumentsParse


class EvaluationShow():

    def __init__(self, task_name, target_path, result_path, config_path=None):
        self.task_name = task_name
        self.config_path = config_path
        self.target_path = target_path
        self.result_path = result_path
        self.color_list = ("#FF0000", "#00FF00", "#0000FF",
                           "#FFFF00", "#00FFFF", "#FF00FF",
                           "#000000", "#054E9F", "#DEB887")

    def roc_show(self):
        config_factory = ConfigFactory()
        task_config = config_factory.get_config(self.task_name, self.config_path)
        if self.task_name.strip() == TaskName.Classify_Task:
            class_number = len(task_config.class_name)
            classify_roc = ClassifyPrecisionAndRecall(class_number)
            class_roc_dict = classify_roc.eval(self.result_path, self.target_path)
            self.save_roc_value(class_roc_dict)
            self.draw_all_roc(class_roc_dict, class_number)

    def save_roc_value(self, class_roc_dict):
        with codecs.open('roc_value.json', 'w', 'utf-8') as f:
            json.dump(class_roc_dict, f, ensure_ascii=False)

    def draw_all_roc(self, roc_value_dict, class_number):
        draw_line = [(0, 1), (0, 1)]
        plt.title('ROC')
        color_count = len(self.color_list)
        for class_index in range(class_number):
            pr_list = roc_value_dict[class_index]
            pecision = [x[0] for x in pr_list]
            recall = [x[1] for x in pr_list]
            color_name = self.color_list[class_index % color_count]
            tag_name = "%d" % class_index
            plt.plot(recall, pecision, linewidth=2, label=tag_name,
                     color=color_name, marker='o')
        plt.plot(draw_line[0], draw_line[1], linewidth=1, color='b')
        plt.legend(loc='upper left')
        plt.xlabel('recall')
        plt.ylabel('pecision')
        plt.xlim(0.0, 1.0)  # set axis limits
        plt.ylim(0.0, 1.0)

        plt.savefig('./ROC.jpg')
        plt.show()


def main():
    print("start...")
    options = ToolArgumentsParse.evaluation_show_path_parse()
    test = EvaluationShow(options.task_name, options.targetPath,
                          options.resultPath, options.config_path)
    test.roc_show()
    print("End of game, have a nice day!")


if __name__ == "__main__":
   main()


