#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import torch
import numpy as np
from easyai.tasks.utility.base_inference import BaseInference
from easyai.name_manager.task_name import TaskName
from easyai.tasks.utility.task_registry import REGISTERED_INFERENCE_TASK


@REGISTERED_INFERENCE_TASK.register_module(TaskName.SuperResolution_Task)
class SuperResolution(BaseInference):

    def __init__(self, model_name, gpu_id, config_path=None):
        super().__init__(model_name, config_path, TaskName.SuperResolution_Task)
        self.set_model_param(data_channel=self.task_config.data['data_channel'],
                             upscale_factor=self.task_config.upscale_factor)
        self.set_model(gpu_id=gpu_id)

    def process(self, input_path, data_type=1, is_show=False):
        pass
        # for i, (oriImg, imgs) in enumerate(dataloader):
        #     img_pil = Image.fromarray(cv2.cvtColor(oriImg, cv2.COLOR_BGR2RGB))
        #     img = img_pil.convert('YCbCr')
        #     y, cb, cr = img.split()
        #     img_to_tensor = ToTensor()
        #     input = img_to_tensor(y).view(1, -1, y.size[1], y.size[0])
        #     # Get detections
        #     with torch.no_grad():
        #         output = self.model(input.to(self.device))[0]
        #
        #     print('Batch %d... Done. (%.3fs)' % (i, time.time() - prev_time))
        #     prev_time = time.time()
        #
        #     out_img_y = output.cpu().detach().numpy()
        #     out_img_y *= 255.0
        #     out_img_y = out_img_y.clip(0, 255)
        #     out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')
        #
        #     out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
        #     out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
        #     out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
        #     show_img = cv2.cvtColor(np.asarray(out_img), cv2.COLOR_RGB2BGR)
        #
        #     cv2.namedWindow("image", 0)
        #     cv2.resizeWindow("image", int(show_img.shape[1] * 0.5), int(show_img.shape[0] * 0.5))
        #     cv2.imshow('image', show_img)
        #
        #     if cv2.waitKey() & 0xFF == 27:
        #         break
    
    def single_image_process(self, input_data):
        pass

    def infer(self, input_data, net_type=0):
        with torch.no_grad():
            image_data = input_data['image'].to(self.device)
            output_list = self.model(image_data)
            prediction = self.compute_output(output_list[:])
        return prediction, output_list

    def postprocess(self, result, threshold=None):
        pass

    def compute_output(self, output_list):
        output = self.common_output(output_list)
        if isinstance(output, (list, tuple)):
            prediction = torch.cat(output, 1)
        else:
            prediction = output
        if prediction is not None:
            prediction = np.squeeze(prediction.data.cpu().numpy())
        return prediction


if __name__ == '__main__':
    pass
