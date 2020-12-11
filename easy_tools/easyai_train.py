#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: lipeijie

import os
from easyai.tools.utility.copy_image import CopyImage
from easyai.train_task import TrainTask
from easyai.base_name.task_name import TaskName
from easyai.config.utility.image_task_config import ImageTaskConfig
from easyai.tools.sample.create_classify_sample import CreateClassifySample
from easyai.tools.sample.create_detection_sample import CreateDetectionSample
from easyai.tools.sample.create_segment_sample import CreateSegmentionSample
from easyai.tools.sample.detection_sample_process import DetectionSampleProcess


class EasyAiModelTrain():

    def __init__(self, train_path, val_path, gpu_id, config_path):
        self.train_path = train_path
        self.val_path = val_path
        self.gpu_id = gpu_id
        self.config_path = config_path
        self.copy_process = CopyImage()
        self.config = ImageTaskConfig()
        self.dataset_path, _ = os.path.split(self.train_path)
        self.images_dir = os.path.join(self.dataset_path, "../JPEGImages")

    def classify_model_train(self, dir_name):
        pretrain_model_path = os.path.join(dir_name, "./data/classnet.pt")
        create_cls_sample = CreateClassifySample()
        create_cls_sample.process_sample(self.images_dir, self.dataset_path, "train_val", 10)
        train_task = TrainTask(TaskName.Classify_Task, self.train_path, self.val_path, True)
        train_task.train('classnet', self.gpu_id, self.config_path, pretrain_model_path)
        save_image_dir = os.path.join(self.config.root_save_dir, "cls_img")
        self.copy_process.copy(self.train_path, save_image_dir)

    def det2d_model_train(self, dir_name):
        pretrain_model_path = os.path.join(dir_name, "./data/detnet.pt")
        create_det2d_sample = CreateDetectionSample()
        create_det2d_sample.createTrainAndTest(self.images_dir, self.dataset_path, 10)
        sample_process = DetectionSampleProcess()
        class_names = sample_process.create_class_names(self.train_path)
        if len(class_names) > 0:
            train_task = TrainTask(TaskName.Detect2d_Task, self.train_path, self.val_path, True)
            train_task.train("denet", self.gpu_id, self.config_path, pretrain_model_path)
            # easy_model_convert(options.task_name, train_task.save_onnx_path)
            save_image_dir = os.path.join(self.config.root_save_dir, "det_img")
            self.copy_process.copy(self.train_path, save_image_dir)
        else:
            print("class name empty!")

    def segment_model_train(self, dir_name):
        pretrain_model_path = os.path.join(dir_name, "./data/segnet.pt")
        cfg_path = os.path.join(dir_name, "./data/segnet.cfg")
        create_seg_sample = CreateSegmentionSample()
        create_seg_sample.create_train_and_test(self.images_dir, self.dataset_path, 10)
        train_task = TrainTask(TaskName.Segment_Task, self.train_path, self.val_path, True)
        train_task.train(cfg_path, self.gpu_id, self.config_path, pretrain_model_path)
        save_image_dir = os.path.join(self.config.root_save_dir, "seg_img")
        self.copy_process.copy(self.train_path, save_image_dir)