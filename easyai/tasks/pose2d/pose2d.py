#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import torch
from easyai.tasks.utility.base_inference import BaseInference
from easyai.tasks.det2d.detect2d_result_process import Detect2dResultProcess
from easyai.tasks.pose2d.pose2d_result_process import Pose2dResultProcess
from easyai.data_loader.pose2d.box2d_dataloader import Box2dLoader
from easyai.visualization.task_show.pose2d_show import Pose2dShow
from easyai.base_name.task_name import TaskName
from easyai.tasks.utility.registry import REGISTERED_INFERENCE_TASK


@REGISTERED_INFERENCE_TASK.register_module(TaskName.Pose2d_Task)
class Pose2d(BaseInference):

    def __init__(self, cfg_path, gpu_id, config_path=None):
        super().__init__(cfg_path, config_path, TaskName.Pose2d_Task)

        self.det_result_process = Detect2dResultProcess(self.task_config.det_config.post_prcoess_type,
                                                        self.task_config.nms_th,
                                                        self.task_config.image_size,
                                                        self.task_config.detect2d_class)
        self.pose_result_process = Pose2dResultProcess(self.task_config.points_count)
        self.model = self.torchModelProcess.create_model(self.model_args, gpu_id)

        self.result_show = Pose2dShow()

    def load_weights(self, weights_path):
        if self.task_config.trian_det:
            self.torchModelProcess.load_latest_model(weights_path[0], self.model.det_model)
            self.model.det_model = self.torchModelProcess.model_test_init(self.model.det_model)
            self.model.det_model.eval()
        if self.task_config.trian_pose:
            self.torchModelProcess.load_latest_model(weights_path[1], self.model.pose_model)
            self.model.pose_model = self.torchModelProcess.model_test_init(self.model.pose_model)
            self.model.pose_model.eval()

    def process(self, input_path, data_type=1, is_show=False):
        dataloader = self.get_image_data_lodaer(input_path)
        image_count = len(dataloader)
        for i, (file_path, src_image, img) in enumerate(dataloader):
            print('%g/%g' % (i + 1, image_count), end=' ')
            self.set_src_size(src_image)

            self.timer.tic()
            prediction = self.infer(img, 0)
            det_objects = self.det_result_process.postprocess(prediction,
                                                              self.src_size,
                                                              self.task_config.det_config.confidence_th)
            objects_pose = []
            if det_objects is None:
                prediction = self.infer(img, 1)
                pose = self.pose_result_process.postprocess(prediction)
                objects_pose.append(pose)
            else:
                box_dataloader = Box2dLoader(det_objects, src_image,
                                             self.task_config.image_size,
                                             self.task_config.data_channel,
                                             self.task_config.resize_type,
                                             self.task_config.normalize_type,
                                             self.task_config.data_mean,
                                             self.task_config.data_std)
                for box, roi_image in box_dataloader:
                    prediction = self.infer(roi_image, 1)
                    pose = self.pose_result_process.postprocess(prediction)
                    objects_pose.append(pose)
            print('Batch %d... Done. (%.3fs)' % (i, self.timer.toc()))

            if not self.result_show.show(src_image, det_objects, objects_pose):
                break

    def infer(self, input_data, net_type=0):
        output = None
        with torch.no_grad():
            if net_type == 0 and self.task_config.trian_det:
                output_list = self.model(input_data.to(self.device), 0)
                output = self.compute_det_output(output_list)
            elif net_type == 1 and self.task_config.trian_pose:
                output_list = self.model(input_data.to(self.device), 1)
                output = self.compute_pose_output(output_list)
        return output

    def compute_det_output(self, output_list):
        count = len(output_list)
        loss_count = len(self.model.det_model.lossList)
        output_count = len(output_list)
        prediction = None
        if loss_count == 1 and output_count == 1:
            prediction = self.model.det_model.lossList[0](output_list[0])
        elif loss_count == 1 and output_count > 1:
            prediction = self.model.det_model.lossList[0](output_list)
        elif loss_count > 1 and loss_count == output_count:
            preds = []
            for i in range(0, count):
                temp = self.model.det_model.lossList[i](output_list[i])
                preds.append(temp)
            prediction = torch.cat(preds, 1)
        else:
            print("compute loss error")
        if prediction is not None:
            prediction = prediction.squeeze(0)
        return prediction

    def compute_pose_output(self, output_list):
        count = len(output_list)
        loss_count = len(self.model.det_model.lossList)
        output_count = len(output_list)
        prediction = None
        if loss_count == 1 and output_count == 1:
            prediction = self.model.pose_model.lossList[0](output_list[0])
        elif loss_count == 1 and output_count > 1:
            prediction = self.model.pose_model.lossList[0](output_list)
        elif loss_count > 1 and loss_count == output_count:
            preds = []
            for i in range(0, count):
                temp = self.model.pose_model.lossList[i](output_list[i])
                preds.append(temp)
            prediction = torch.cat(preds, 1)
        else:
            print("compute loss error")
        if prediction is not None:
            prediction = prediction.squeeze(0)
        return prediction
