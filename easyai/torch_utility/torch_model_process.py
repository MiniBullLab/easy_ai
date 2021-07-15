#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import pathlib
import os.path
import re
import torch
from collections import OrderedDict
from easyai.torch_utility.torch_device_process import TorchDeviceProcess
from easyai.model.utility.model_factory import ModelFactory, ModelWeightInit
from easyai.utility.logger import EasyLogger


class TorchModelProcess():

    def __init__(self):
        self.torchDeviceProcess = TorchDeviceProcess()
        self.modelFactory = ModelFactory()
        self.model_weight_init = ModelWeightInit()

        self.torchDeviceProcess.initTorch()
        self.best_value = -1
        self.is_multi_gpu = False

    def create_model(self, model_config, gpu_id, is_multi_gpu=False):
        self.is_multi_gpu = is_multi_gpu
        self.torchDeviceProcess.setGpuId(gpu_id)
        model = self.modelFactory.get_model(model_config)
        return model

    def init_model(self, model, init_type):
        self.model_weight_init.set_init_type(init_type)
        self.model_weight_init.init_weight(model)

    def load_pretain_model(self, weight_path, model):
        if weight_path is not None:
            if os.path.exists(weight_path):
                EasyLogger.debug("Loading pretainModel from {}".format(weight_path))
                model_dict = model.state_dict()
                checkpoint = torch.load(weight_path)
                pretrained_dict = checkpoint['model']
                # pretrained_dict = self.filter_param_dict(pretrained_dict)
                new_pretrained_dict = {}
                for k, v in pretrained_dict.items():
                    if k in model_dict and v.shape == model_dict[k].shape:
                        new_pretrained_dict[k] = v
                EasyLogger.debug("Load pretrained parameters:")
                for k, v in new_pretrained_dict.items():
                    EasyLogger.debug("{} {}".format(k, v.shape))
                model_dict.update(new_pretrained_dict)
                model.load_state_dict(model_dict)
                EasyLogger.warn("Load pretrained parameters(%s) success" % weight_path)
            else:
                EasyLogger.warn("pretrained model %s not exist" % weight_path)

    def load_latest_model(self, weight_path, model, dict_name="model"):
        count = self.torchDeviceProcess.getCUDACount()
        checkpoint = None
        if os.path.exists(weight_path):
            try:
                if count > 1:
                    checkpoint = torch.load(weight_path, map_location=torch.device("cpu"))
                    state = self.convert_state_dict(checkpoint[dict_name])
                    model.load_state_dict(state)
                else:
                    checkpoint = torch.load(weight_path, map_location=torch.device("cpu"))
                    model.load_state_dict(checkpoint[dict_name])
            except Exception as err:
                # os.remove(weight_path)
                checkpoint = None
                EasyLogger.warn(err)
        else:
            EasyLogger.error("Latest model %s exists" % weight_path)
        result = self.get_latest_model_value(checkpoint)
        return result

    def get_latest_model_value(self, checkpoint):
        start_epoch = 0
        value = -1
        if checkpoint is not None and len(checkpoint) > 0:
            if checkpoint.get('epoch') is not None:
                start_epoch = checkpoint['epoch'] + 1
            if checkpoint.get('best_value') is not None:
                value = checkpoint['best_value']
        self.set_model_best_value(value)
        return start_epoch, value

    def save_latest_model(self, epoch, best_value, model, weights_path):
        # Save latest checkpoint
        checkpoint = {'epoch': epoch,
                      'best_value': best_value,
                      'model': model.state_dict()
                      }
        torch.save(checkpoint, weights_path)
        # scripted_module = torch.jit.script(model)
        # temp_path = pathlib.Path(weights_path)
        # temp_path = temp_path.with_suffix(".native")
        # torch.jit.save(scripted_module, ocr(temp_path))

    def save_best_model(self, value, latest_weights_file, best_weights_file):
        if value >= self.best_value:
            self.best_value = value
            os.system('cp {} {}'.format(
                latest_weights_file,
                best_weights_file,
            ))
            # temp_path = pathlib.Path(latest_weights_file)
            # temp_path = temp_path.with_suffix(".native")
            # best_temp_path = pathlib.Path(best_weights_file)
            # best_temp_path = best_temp_path.with_suffix(".native")
            # os.system('cp {} {}'.format(
            #     ocr(temp_path),
            #     ocr(best_temp_path),
            # ))
        return self.best_value

    def load_latest_optimizer(self, optimizer_path, optimizer, amp_opt=None):
        if os.path.exists(optimizer_path):
            try:
                checkpoint = torch.load(optimizer_path)
                optimizer.load_state_dict(checkpoint['optimizer'])
                if amp_opt is not None:
                    amp_opt.load_state_dict(checkpoint['amp'])
            except Exception as err:
                os.remove(optimizer_path)
                EasyLogger.warn(err)
        else:
            EasyLogger.warn("Loading optimizer %s fail" % optimizer_path)

    def save_optimizer_state(self, optimizer_save_path,
                             epoch, optimizer, amp_opt=None):
        checkpoint = {'epoch': epoch,
                      'optimizer': optimizer.state_dict()
                      }
        if amp_opt is not None:
            checkpoint['amp'] = amp_opt.state_dict()
        torch.save(checkpoint, optimizer_save_path)

    def load_latest_list_optimizer(self, optimizer_path, optimizer_list):
        if os.path.exists(optimizer_path):
            checkpoint = torch.load(optimizer_path)
            for optimizer_name in checkpoint:
                str_list = optimizer_name.spilt('_')
                if str_list[0].strip() == 'optimizer':
                    index = int(str_list[1])
                    optimizer_list[index].load_state_dict(checkpoint[optimizer_name])
        else:
            EasyLogger.warn("Loading optimizer %s fail" % optimizer_path)

    def save_list_optimizer_state(self, optimizer_save_path,
                                  epoch, optimizer_list):
        checkpoint = {'epoch': epoch}
        for index, optimizer in enumerate(optimizer_list):
            checkpoint['optimizer_%d' % index] = optimizer.state_dict()
        torch.save(checkpoint, optimizer_save_path)

    def set_model_best_value(self, value):
        self.best_value = value

    def model_train_init(self, model):
        count = self.torchDeviceProcess.getCUDACount()
        if count > 1 and self.is_multi_gpu:
            EasyLogger.debug('Using %d GPUs' % count)
            model = torch.nn.DataParallel(model)
        model = model.to(self.torchDeviceProcess.device)
        return model

    def model_test_init(self, model):
        model = model.to(self.torchDeviceProcess.device)
        return model

    def model_clip_grad(self, model):
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)

    def convert_state_dict(self, state_dict):
        """Converts a state dict saved from a dataParallel module to normal
           module state_dict inplace
           :param state_dict is the loaded DataParallel model_state

        """
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        return new_state_dict

    def get_device(self):
        return self.torchDeviceProcess.device

    def filter_param_dict(self, state_dict: dict, include: str = None, exclude: str = None):
        assert isinstance(state_dict, dict)
        include_re = None
        if include is not None:
            include_re = re.compile(include)
        exclude_re = None
        if exclude is not None:
            exclude_re = re.compile(exclude)
        res_dict = {}
        for k, p in state_dict.items():
            if include_re is not None:
                if include_re.match(k) is None:
                    continue
            if exclude_re is not None:
                if exclude_re.match(k) is not None:
                    continue
            res_dict[k] = p
        return res_dict
