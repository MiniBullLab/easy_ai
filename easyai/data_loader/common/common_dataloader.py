#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import torch.utils.data as data
from easyai.name_manager.dataloader_name import DataloaderName
from easyai.data_loader.utility.dataloader_registry import REGISTERED_TRAIN_DATALOADER
from easyai.data_loader.utility.dataloader_registry import REGISTERED_VAL_DATALOADER


REGISTERED_TRAIN_DATALOADER.add_module(data.DataLoader, DataloaderName.DataLoader)
REGISTERED_VAL_DATALOADER.add_module(data.DataLoader, DataloaderName.DataLoader)
