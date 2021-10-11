#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import random
import pickle
import numpy as np
from scipy.spatial.distance import mahalanobis
from easyai.tasks.utility.base_post_process import BasePostProcess
from easyai.name_manager.post_process_name import PostProcessName
from easyai.tasks.utility.task_registry import REGISTERED_POST_PROCESS
from easyai.utility.logger import EasyLogger


@REGISTERED_POST_PROCESS.register_module(PostProcessName.PadimPostProcess)
class PadimPostProcess(BasePostProcess):

    def __init__(self, save_path, threshold,
                 feature_dimension=1536, select_dimension=153):
        super().__init__()
        self.save_path = save_path
        self.threshold = threshold
        self.select_dimension = select_dimension
        self.embedding_list = []
        random.seed(1024)
        self.select_index = random.sample(range(0, feature_dimension),
                                          self.select_dimension)

    def reset(self):
        self.embedding_list = []

    def add_embedding(self, prediction):
        self.embedding_list.extend(np.array(prediction))

    def save_embedding(self):
        total_embeddings = np.array(self.embedding_list)
        embedding_vectors = total_embeddings[:, self.select_index, :, :]
        # calculate multivariate Gaussian distribution
        batch, channel, height, width = embedding_vectors.shape
        embedding_vectors = embedding_vectors.reshape(batch, channel, -1)
        mean = np.mean(embedding_vectors, axis=0)
        cov = np.zeros([channel, channel, height * width], dtype=np.float32)
        I = np.identity(channel)
        for i in range(height * width):
            cov[:, :, i] = np.cov(embedding_vectors[:, :, i], rowvar=False) + 0.01 * I

        train_outputs = [mean, cov]
        with open(self.save_path, 'wb') as f:
            pickle.dump(train_outputs, f)

    def __call__(self, prediction):
        embeddings = pickle.load(open(self.save_path, 'rb'))
        if prediction.ndim == 3:
            prediction = np.expand_dims(prediction, 0)
        embedding_vectors = prediction[:, self.select_index, :, :]
        EasyLogger.debug("embedding_vectors: {}".format(embedding_vectors.shape))
        # calculate distance matrix
        batch, channel, height, width = embedding_vectors.shape
        embedding_vectors = embedding_vectors.reshape(batch, channel, height * width)
        dist_list = []
        for i in range(height * width):
            mean = embeddings[0][:, i]
            conv_inv = np.linalg.inv(embeddings[1][:, :, i])
            dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
            dist_list.append(dist)

        score = max(np.array(dist_list).flatten())
        EasyLogger.debug("score: {}".format(score))

        if score > self.threshold:
            class_index = 1
        else:
            class_index = 0
        return class_index, score
