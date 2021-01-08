#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import json
import codecs
import matplotlib.pyplot as plt
from easyai.helper.arguments_parse import ToolArgumentsParse
from easyai.evaluation.classify_pr import ClassifyPrecisionAndRecall


class EvaluationShow():

    def __init__(self, target_path, result_path):
        self.target_path = target_path
        self.result_path = result_path
        self.class_number = 2
        self.color_list = ["r", "b"]

    def roc_show(self):
        classify_roc = ClassifyPrecisionAndRecall(self.class_number)
        class_roc_dict = classify_roc.eval(self.result_path, self.target_path)
        self.save_roc_value(class_roc_dict)

    def save_roc_value(self, class_roc_dict):
        with codecs.open('roc_value.json', 'w', 'utf-8') as f:
            json.dump(class_roc_dict, f, ensure_ascii=False)

    def draw_all_roc(self, roc_value_dict):
        draw_line = [(0, 1), (0, 1)]
        plt.title('ROC')
        for class_index in range(self.class_number):
            pr_list = roc_value_dict[class_index]
            pecision = [x[0] for x in pr_list]
            recall = [x[1] for x in pr_list]
            color_name = self.color_list[class_index]
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
    options = ToolArgumentsParse.roc_show_path_parse()
    test = EvaluationShow(options.targetPath, options.resultPath)
    test.roc_show()
    print("End of game, have a nice day!")


if __name__ == "__main__":
   main()


