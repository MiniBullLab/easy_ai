#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import traceback
import os
import torch
from easyai.tasks.utility.base_test import BaseTest
from easyai.name_manager.task_name import TaskName
from easyai.tasks.utility.task_registry import REGISTERED_TEST_TASK
from easyai.utility.registry import build_from_cfg
from easy_tracking.utility.tracking_registry import REGISTERED_REID
from easyai.utility.logger import EasyLogger


@REGISTERED_TEST_TASK.register_module(TaskName.Det2D_REID_TASK)
class Det2dReidTest(BaseTest):

    def __init__(self, model_name, gpu_id, config_path=None):
        super().__init__(TaskName.Det2D_REID_TASK)
        self.inference = self.build_reid_task(TaskName.Det2D_REID_TASK,
                                              gpu_id, config_path)
        self.set_test_config(self.inference.task_config)
        self.set_model()

    def process_test(self, val_path, epoch=0):
        self.create_dataloader(val_path)
        if not self.start_test():
            EasyLogger.info("no test!")
            return
        mAP = self.test(epoch)
        print("mAP: {:.5f}".format(mAP))

    def test(self, epoch=0):
        os.system('rm -rf ' + self.test_task_config.save_result_dir)
        os.makedirs(self.test_task_config.save_result_dir, exist_ok=True)
        for i, batch_data in enumerate(self.dataloader):
            result, output_list = self.inference.single_image_process(batch_data)
            with torch.no_grad():
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

    def build_reid_task(self, reid_task_name, gpu_id, config_path):
        reid_task = None
        reid_task_config = {"type": reid_task_name.strip(),
                            "model_name": None,
                            "gpu_id": gpu_id,
                            "config_path": config_path}
        try:
            if REGISTERED_REID.has_class(reid_task_name.strip()):
                reid_task = build_from_cfg(reid_task_config, REGISTERED_REID)
            else:
                EasyLogger.error("%s reid task not exits" % reid_task_name)
        except ValueError as err:
            EasyLogger.error(traceback.format_exc())
            EasyLogger.error(err)
        except TypeError as err:
            EasyLogger.error(traceback.format_exc())
            EasyLogger.error(err)
        return reid_task

