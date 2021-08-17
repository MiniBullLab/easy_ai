#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import random
import torch
import pickle
import numpy as np
from random import sample
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
import torch.nn.functional as F
from easyai.tasks.utility.base_post_process import BasePostProcess
from easyai.name_manager.post_process_name import PostProcessName
from easyai.tasks.utility.task_registry import REGISTERED_POST_PROCESS


@REGISTERED_POST_PROCESS.register_module(PostProcessName.PadimPostProcess)
class PadimPostProcess(BasePostProcess):

    def __init__(self, save_path, threshold,
                 feature_dimension=1792, select_dimension=550):
        super().__init__()
        self.save_path = save_path
        self.threshold = threshold
        self.select_dimension = select_dimension
        self.embedding_list = []
        random.seed(1024)
        self.select_index = torch.tensor(sample(range(0, feature_dimension),
                                                self.select_dimension))

    def reset(self):
        self.embedding_list = []

    def add_embedding(self, prediction):
        self.embedding_list.extend(np.array(prediction))

    def save_embedding(self):
        embedding_vectors = torch.tensor(np.array(self.embedding_list))
        embedding_vectors = torch.index_select(embedding_vectors, 1, self.select_index)
        # calculate multivariate Gaussian distribution
        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W)
        mean = torch.mean(embedding_vectors, dim=0).numpy()
        cov = torch.zeros(C, C, H * W).numpy()
        I = np.identity(C)
        for i in range(H * W):
            cov[:, :, i] = np.cov(embedding_vectors[:, :, i].numpy(), rowvar=False) + 0.01 * I
        # save learned distribution
        train_outputs = [mean, cov]
        with open(self.save_path, 'wb') as f:
            pickle.dump(train_outputs, f)

    def __call__(self, prediction):
        embeddings = pickle.load(open(self.save_path, 'rb'))
        if prediction.ndim == 3:
            prediction = np.expand_dims(prediction, 0)
        prediction = torch.tensor(prediction)
        embedding_vectors = torch.index_select(prediction, 1, self.select_index)
        # calculate distance matrix
        batch, channel, height, width = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(batch, channel, height * width).numpy()
        dist_list = []
        for i in range(height * width):
            mean = embeddings[0][:, i]
            conv_inv = np.linalg.inv(embeddings[1][:, :, i])
            dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
            dist_list.append(dist)

        dist_list = np.array(dist_list).transpose(1, 0).reshape(batch, height, width)

        dist_list = torch.tensor(dist_list)
        score_map = F.interpolate(dist_list.unsqueeze(1), scale_factor=4, mode='bilinear',
                                  align_corners=False).squeeze().numpy()

        # apply gaussian smoothing on the score map
        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=4)

        scores = score_map.reshape(score_map.shape[0], -1).max(axis=1)

        if scores[0] > self.threshold:
            class_index = 1
        else:
            class_index = 0
        return class_index, scores[0]
