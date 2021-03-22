#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import torch
from easyai.tasks.utility.base_inference import BaseInference
from easyai.tasks.pose2d.pose2d_result_process import Pose2dResultProcess
from easyai.visualization.task_show.pose2d_show import Pose2dShow
from easyai.base_name.task_name import TaskName
from easyai.tasks.utility.registry import REGISTERED_INFERENCE_TASK


@REGISTERED_INFERENCE_TASK.register_module(TaskName.Pose2d_Task)
class Pose2d(BaseInference):

    def __init__(self, cfg_path, gpu_id, config_path=None):
        super().__init__(cfg_path, config_path, TaskName.Pose2d_Task)

        self.pose_result_process = Pose2dResultProcess(self.task_config.post_prcoess_type,
                                                       self.task_config.points_count,
                                                       self.task_config.image_size)
        self.model = self.torchModelProcess.create_model(self.model_args, gpu_id)

        self.result_show = Pose2dShow()

    def process(self, input_path, data_type=1, is_show=False):
        dataloader = self.get_image_data_lodaer(input_path)
        image_count = len(dataloader)
        for i, (file_path, src_image, img) in enumerate(dataloader):
            print('%g/%g' % (i + 1, image_count), end=' ')
            self.timer.tic()
            self.set_src_size(src_image)
            objects_pose = self.single_image_process(self.src_size, img)
            print('Batch %d... Done. (%.3fs)' % (i, self.timer.toc()))

            if not self.result_show.show(src_image, (), objects_pose):
                break

    def single_image_process(self, src_size, input_image):
        objects_pose = []
        prediction, _ = self.infer(input_image)
        pose = self.pose_result_process.postprocess(prediction, src_size, 0.4)
        objects_pose.append(pose)
        return objects_pose

    def infer(self, input_data, net_type=0):
        with torch.no_grad():
            output_list = self.model(input_data.to(self.device))
            output = self.compute_output(output_list)
        return output, output_list

    def compute_output(self, output_list):
        count = len(output_list)
        loss_count = len(self.model.lossList)
        output_count = len(output_list)
        prediction = None
        if loss_count == 1 and output_count == 1:
            prediction = self.model.lossList[0](output_list[0])
        elif loss_count == 1 and output_count > 1:
            prediction = self.model.lossList[0](output_list)
        elif loss_count > 1 and loss_count == output_count:
            preds = []
            for i in range(0, count):
                temp = self.model.lossList[i](output_list[i])
                preds.append(temp)
            prediction = torch.cat(preds, 1)
        else:
            print("compute loss error")
        if prediction is not None:
            prediction = prediction.squeeze(0)
            prediction = prediction.detach().numpy()
        return prediction