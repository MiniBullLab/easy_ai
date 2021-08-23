#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
from easyai.tasks.utility.base_test import BaseTest
from easyai.tasks.det2d.detect2d import Detection2d
from easyai.name_manager.task_name import TaskName
from easyai.tasks.utility.task_registry import REGISTERED_TEST_TASK
from easyai.utility.logger import EasyLogger


@REGISTERED_TEST_TASK.register_module(TaskName.Detect2d_Task)
class Detection2dTest(BaseTest):

    def __init__(self, model_name, gpu_id, config_path=None):
        super().__init__(TaskName.Detect2d_Task)
        self.inference = Detection2d(model_name, gpu_id, config_path)
        self.set_test_config(self.inference.task_config)
        self.set_model()
        self.inference.result_process.set_threshold(5e-3)

    def process_test(self, val_path, epoch=0):
        self.create_dataloader(val_path)
        if not self.start_test():
            EasyLogger.info("no test!")
            return
        mAP = self.test(epoch)
        print("mAP: {.5f}".format(mAP))

    def test(self, epoch=0):
        os.system('rm -rf ' + self.test_task_config.save_result_dir)
        os.makedirs(self.test_task_config.save_result_dir, exist_ok=True)
        for i, batch_data in enumerate(self.dataloader):
            result, output_list = self.inference.single_image_process(batch_data['src_size'][0].numpy(),
                                                                      batch_data)
            loss_value = self.compute_loss(output_list, batch_data)
            self.metirc_loss(i, loss_value)
            self.print_test_info(i, loss_value)
            self.inference.save_result(batch_data['image_path'][0], result, 1)

        mAP, aps = self.evaluation.eval(self.test_task_config.save_result_dir,
                                        self.val_path)
        self.save_test_value(epoch, mAP, aps)
        return mAP

    def save_test_value(self, epoch, mAP, aps):
        # Write epoch results
        with open(self.test_task_config.evaluation_result_path, 'a') as file:
            # file.write('%11.3g' * 2 % (mAP, aps[0]) + '\n')
            file.write("Epoch: {} | mAP: {:.3f} | ".format(epoch, mAP))
            for i, ap in enumerate(aps):
                file.write(self.test_task_config.detect2d_class[i] + ": {:.3f} ".format(ap))
            file.write("\n")

