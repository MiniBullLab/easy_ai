#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
from easyai.tasks.utility.base_test import BaseTest
from easyai.tasks.multi_task.det2d_seg_task import Det2dSegTask
from easyai.name_manager.evaluation_name import EvaluationName
from easyai.name_manager.task_name import TaskName
from easyai.tasks.utility.task_registry import REGISTERED_TEST_TASK
from easyai.utility.logger import EasyLogger


@REGISTERED_TEST_TASK.register_module(TaskName.Det2d_Seg_Task)
class Det2dSegTaskTest(BaseTest):

    def __init__(self, model_name, gpu_id, config_path=None):
        super().__init__(TaskName.Det2d_Seg_Task)
        self.inference = Det2dSegTask(model_name, gpu_id, config_path)
        self.set_test_config(self.inference.task_config)
        self.set_model()
        self.evaluation_args = {"type": EvaluationName.DetectionMeanAp,
                                'class_names': self.test_task_config.detect2d_class}
        self.det2d_evaluator = self.evaluation_factory.get_evaluation(self.evaluation_args)
        self.evaluation_args = {"type": EvaluationName.SegmentionMetric,
                                'num_class': len(self.test_task_config.segment_name)}
        self.seg_metric = self.evaluation_factory.get_evaluation(self.evaluation_args)
        self.threshold_det = 5e-3
        self.threshold_seg = 0.5

    def process_test(self, val_path, epoch=0):
        self.create_dataloader(val_path)
        if not self.start_test():
            EasyLogger.info("no test!")
            return
        self.test(epoch)

    def test(self, epoch=0):
        os.system('rm -rf ' + self.test_task_config.save_result_dir)
        os.makedirs(self.test_task_config.save_result_dir, exist_ok=True)
        self.seg_metric.reset()
        for i, (image_path, src_image, input_image, segment_targets) in enumerate(self.dataloader):
            self.multi_task_inference.set_src_size(src_image.numpy()[0])

            predict_dets, predict_seg = self.multi_task_inference.infer(input_image)
            detection_objects, _, result_seg = self.multi_task_inference.postprocess((predict_dets,
                                                                                      predict_seg),
                                                                                     (self.threshold_det,
                                                                                      self.threshold_seg))

            gt = segment_targets[0].data.cpu().numpy()
            self.seg_metric.numpy_eval(result_seg, gt)

            self.print_test_info(i)

            path, filename_post = os.path.split(image_path[0])
            self.multi_task_inference.save_result(filename_post, detection_objects)

        mAP, aps = self.det2d_evaluator.eval(self.test_task_config.save_result_dir, self.val_path)
        score, class_score = self.seg_metric.get_score()
        self.save_test_value(epoch, mAP, aps, score, class_score)
        return mAP, score['Mean IoU']

    def save_test_value(self, epoch, mAP, aps, score, class_score):
        # Write epoch results
        with open(self.test_task_config.evaluation_result_path, 'a') as file:
            # file.write('%11.3g' * 2 % (mAP, aps[0]) + '\n')
            file.write("Epoch: {} | mAP: {:.3f} | mIoU: {:.3f} | ".format(epoch, mAP, score['Mean IoU : \t']))
            for i, ap in enumerate(aps):
                file.write(self.test_task_config.detect_name[i] + ": {:.3f} ".format(ap))
            for i, iou in class_score.items():
                file.write(self.test_task_config.segment_name[i][0] + ": {:.3f} ".format(iou))
            file.write("\n")

