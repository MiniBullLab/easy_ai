#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.tasks.utility.base_inference import BaseInference
from easyai.tasks.det2d.detect2d import Detection2d
from easyai.tasks.pose2d.pose2d import Pose2d
from easyai.data_loader.pose2d.box2d_dataloader import Box2dLoader
from easyai.visualization.task_show.det_pose2d_show import DetAndPose2dShow
from easyai.base_name.task_name import TaskName
from easyai.tasks.utility.registry import REGISTERED_INFERENCE_TASK


@REGISTERED_INFERENCE_TASK.register_module(TaskName.DetPose2d_Task)
class DetPose2dTask(BaseInference):

    def __init__(self, cfg_path, gpu_id, config_path=None):
        super().__init__(cfg_path, config_path, TaskName.DetPose2d_Task)
        model_name_list = cfg_path.split("|")
        self.det2d_inference = Detection2d(model_name_list[0], gpu_id)
        self.pose2d_inference = Pose2d(model_name_list[0], gpu_id)
        self.result_show = DetAndPose2dShow()

    def load_weights(self, weights_path):
        weights_path_list = weights_path.split("|")
        self.det2d_inference.load_weights(weights_path_list[0])
        self.pose2d_inference.load_weights(weights_path_list[1])

    def process(self, input_path, data_type=1, is_show=False):
        dataloader = self.get_image_data_lodaer(input_path)
        image_count = len(dataloader)
        for i, (file_path, src_image, img) in enumerate(dataloader):
            print('%g/%g' % (i + 1, image_count), end=' ')
            self.timer.tic()
            self.set_src_size(src_image)
            detection_objects = self.det2d_inference.single_image_process(self.src_size, img)
            box_dataloader = Box2dLoader(detection_objects, src_image,
                                         self.pose2d_inference.task_config.image_size,
                                         self.pose2d_inference.task_config.data_channel,
                                         self.pose2d_inference.task_config.resize_type,
                                         self.pose2d_inference.task_config.normalize_type,
                                         self.pose2d_inference.task_config.data_mean,
                                         self.pose2d_inference.task_config.data_std)
            objects_pose = []
            for box, roi_image in box_dataloader:
                pose = self.pose2d_inference.single_image_process((box.width(), box.height()),
                                                                  roi_image)
                for point in pose.get_key_points():
                    point.x = point.x + box.min_corner.x
                    point.y = point.y + box.min_corner.y
                objects_pose.append(pose)
            print('Batch %d... Done. (%.3fs)' % (i, self.timer.toc()))

            if not self.result_show.show(src_image, detection_objects, objects_pose,
                                         self.pose2d_inference.task_config.skeleton):
                break

    def infer(self, input_data, net_type=0):
        pass

