#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: lipeijie

import traceback
from easyai.utility.logger import EasyLogger
import os
from easyai.tools.utility.copy_image import CopyImage
from easyai.train_task import TrainTask
from easyai.test_task import TestTask
from easyai.name_manager.task_name import TaskName
from easyai.tools.sample_tool.create_classify_sample import CreateClassifySample
from easyai.tools.sample_tool.create_detection_sample import CreateDetectionSample
from easyai.tools.sample_tool.create_segment_sample import CreateSegmentionSample
from easyai.tools.sample_tool.create_rec_text_sample import CreateRecognizeTextSample
from easyai.tools.sample_tool.sample_info_get import SampleInformation
from easyai.tools.feature_save.one_class_feature_save import OneClassFeatureSave
from easy_tools.model_train.arm_config import ARMConfig
from easy_tools.model_train.segnet_process import SegNetProcess


class EasyAiModelTrain():

    def __init__(self, train_path, val_path, gpu_id):
        self.train_path = train_path
        self.val_path = val_path
        self.gpu_id = gpu_id
        self.copy_process = CopyImage()
        self.dataset_path, _ = os.path.split(self.train_path)
        self.images_dir = os.path.join(self.dataset_path, "../JPEGImages")
        self.sample_process = SampleInformation()
        self.arm_config = ARMConfig()

    def binary_classidy_model_train(self, dir_name):
        input_name = ['ng_input']
        output_name = ['ng_output']
        pretrain_model_path = os.path.join(dir_name, "./data/classnet.pt")
        create_cls_sample = CreateClassifySample()
        create_cls_sample.process_sample(self.images_dir, self.dataset_path, "train_val", 10)
        class_names = self.sample_process.create_class_names(self.train_path,
                                                             TaskName.Classify_Task)
        if len(class_names) == 2:
            try:
                train_task = TrainTask(TaskName.Classify_Task, self.train_path, self.val_path)
                train_task.set_convert_param(True, input_name, output_name)
                train_task.train('binarynet', self.gpu_id, None, pretrain_model_path)
                save_image_dir = os.path.join(EasyLogger.ROOT_DIR, "binary_cls_img")
                self.copy_process.copy(self.train_path, save_image_dir)
            except Exception as err:
                EasyLogger.error(traceback.format_exc())
                EasyLogger.error(err)
        else:
            EasyLogger.warn("binary classify class name error!")

    def classify_model_train(self, dir_name):
        input_name = ['cls_input']
        output_name = ['cls_output']
        pretrain_model_path = os.path.join(dir_name, "./data/classnet.pt")
        create_cls_sample = CreateClassifySample()
        create_cls_sample.process_sample(self.images_dir, self.dataset_path, "train_val", 10)
        class_names = self.sample_process.create_class_names(self.train_path,
                                                             TaskName.Classify_Task)
        if len(class_names) > 1:
            try:
                train_task = TrainTask(TaskName.Classify_Task, self.train_path, self.val_path)
                train_task.set_convert_param(True, input_name, output_name)
                train_task.train('classnet', self.gpu_id, None, pretrain_model_path)
                save_image_dir = os.path.join(EasyLogger.ROOT_DIR, "cls_img")
                self.copy_process.copy(self.train_path, save_image_dir)
                self.arm_config.create_classnet_config(input_name, output_name,
                                                       class_names)
            except Exception as err:
                EasyLogger.error(traceback.format_exc())
                EasyLogger.error(err)
        else:
            EasyLogger.warn("classify class name empty!")

    def det2d_model_train(self, dir_name):
        input_name = ['data']
        output_name = ['det_output0', 'det_output1', 'det_output2']
        pretrain_model_path = os.path.join(dir_name, "./data/detnet.pt")
        create_det2d_sample = CreateDetectionSample()
        create_det2d_sample.create_train_and_test(self.images_dir, self.dataset_path, 10)
        class_names = self.sample_process.create_class_names(self.train_path,
                                                             TaskName.Detect2d_Task)
        if len(class_names) > 0:
            try:
                train_task = TrainTask(TaskName.Detect2d_Task, self.train_path, self.val_path)
                train_task.set_convert_param(True, input_name, output_name)
                train_task.train("denet", self.gpu_id, None, pretrain_model_path)
                # easy_model_convert(options.task_name, train_task.save_onnx_path)
                save_image_dir = os.path.join(EasyLogger.ROOT_DIR, "det_img")
                self.copy_process.copy(self.train_path, save_image_dir)
                self.arm_config.create_denet_config(input_name, output_name,
                                                    class_names)
            except Exception as err:
                EasyLogger.error(traceback.format_exc())
                EasyLogger.error(err)
        else:
            EasyLogger.warn("det2d class name empty!")

    def segment_model_train(self, dir_name):
        input_name = ['seg_input']
        output_name = ['seg_output']
        try:
            segnet_process = SegNetProcess()
            segnet_process.png_process(self.train_path)
            pretrain_model_path = os.path.join(dir_name, "./data/segnet.pt")
            create_seg_sample = CreateSegmentionSample()
            create_seg_sample.create_train_and_test(self.images_dir, self.dataset_path, 10)

            segnet_process.resize_process(self.train_path)
            segnet_process.resize_process(self.val_path)

            train_task = TrainTask(TaskName.Segment_Task, self.train_path, self.val_path)
            train_task.set_convert_param(True, input_name, output_name)
            train_task.train("segnet", self.gpu_id, None, pretrain_model_path)
            save_image_dir = os.path.join(EasyLogger.ROOT_DIR, "seg_img")
            self.copy_process.copy(self.train_path, save_image_dir)
            self.arm_config.create_segnet_config(input_name, output_name)
        except Exception as err:
            EasyLogger.error(traceback.format_exc())
            EasyLogger.error(err)

    def rec_text_model_train(self, dir_name):
        input_name = ['text_input']
        output_name = ['text_output']
        pretrain_model_path = os.path.join(dir_name, "./data/TextNet.pt")
        try:
            create_sample = CreateRecognizeTextSample()
            create_sample.create_train_and_test(self.images_dir, self.dataset_path, 8,
                                                language=("english",))

            train_task = TrainTask(TaskName.RecognizeText, self.train_path, self.val_path)
            train_task.set_convert_param(True, input_name, output_name)
            train_task.train("TextNet", self.gpu_id, None, pretrain_model_path)
            save_image_dir = os.path.join(EasyLogger.ROOT_DIR, "text_img")
            self.copy_process.copy(self.train_path, save_image_dir, '|', 10)
            self.arm_config.create_textnet_config(input_name, output_name)
        except Exception as err:
            EasyLogger.error(traceback.format_exc())
            EasyLogger.error(err)

    def one_class_model_train(self, dir_name):
        input_name = ['one_class_input']
        output_name = ['one_class_output']
        model_path = os.path.join(dir_name, "./data/OneClassNet.pt")
        config_path = os.path.join(dir_name, "./data/OneClass.config")
        try:
            task = OneClassFeatureSave("OneClassNet", 0, config_path)
            task.load_weights(model_path)
            task.process_test(self.train_path)
            test_task = TestTask(TaskName.OneClass, self.val_path)
            test_task.set_convert_param(True, input_name, output_name)
            test_task.test("OneClassNet", 0, model_path, config_path)
            save_image_dir = os.path.join(EasyLogger.ROOT_DIR, "one_class_img")
            self.copy_process.copy(self.train_path, save_image_dir)
        except Exception as err:
            EasyLogger.error(traceback.format_exc())
            EasyLogger.error(err)
