#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie


from torch import nn


class ModelWeightInit():

    def __init__(self):
        self.init_type = "kaiming"

    def set_init_type(self, init_type):
        self.init_type = init_type

    def init_weight(self, model):
        # weight initialization
        if model is None:
            return
        if self.init_type == "kaiming":
            self.kaiming_init(model)
        else:
            self.utility_init(model, self.init_type)

    def kaiming_init(self, model, scale=1):
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def utility_init(self, model, init_type='normal', gain=0.02):
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d,
                              nn.ConvTranspose2d, nn.ConvTranspose3d,
                              nn.Linear)):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                    if m.bias is not None:
                        nn.init.normal_(m.bias.data)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                    if m.bias is not None:
                        nn.init.normal_(m.bias.data)
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                    if m.bias is not None:
                        nn.init.normal_(m.bias.data)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d,
                                nn.BatchNorm3d)):
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.normal_(m.bias.data)
            elif isinstance(m, nn.ConvTranspose1d):
                nn.init.normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.normal_(m.bias.data)
            elif isinstance(m, nn.LSTM):
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        nn.init.orthogonal_(param.data)
                    else:
                        nn.init.normal_(param.data)
            elif isinstance(m, nn.LSTMCell):
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        nn.init.orthogonal_(param.data)
                    else:
                        nn.init.normal_(param.data)
            elif isinstance(m, nn.GRU):
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        nn.init.orthogonal_(param.data)
                    else:
                        nn.init.normal_(param.data)
            elif isinstance(m, nn.GRUCell):
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        nn.init.orthogonal_(param.data)
                    else:
                        nn.init.normal_(param.data)
        print('initialize network with %s' % init_type)

