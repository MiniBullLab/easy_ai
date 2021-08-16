#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import torch
import os
import pickle
import numpy as np
from sklearn.random_projection import SparseRandomProjection
from sklearn.neighbors import NearestNeighbors
from easyai.tasks.one_class.post_process.kcenter_greedy import kCenterGreedy
from easyai.tasks.one_class.post_process.KNN import KNN
from easyai.tasks.utility.base_post_process import BasePostProcess
from easyai.name_manager.post_process_name import PostProcessName
from easyai.tasks.utility.task_registry import REGISTERED_POST_PROCESS


@REGISTERED_POST_PROCESS.register_module(PostProcessName.PatchCorePostProcess)
class PatchCorePostProcess(BasePostProcess):

    def __init__(self, save_path, threshold,
                 sampling_ratio=0.01, neighbor_count=9):
        super().__init__()
        self.save_path = save_path
        self.threshold = threshold
        self.sampling_ratio = sampling_ratio
        self.neighbor_count = neighbor_count
        self.method = ""
        self.embedding_list = []
        # Random projection
        # 'auto' => Johnson-Lindenstrauss lemma
        self.randomprojector = SparseRandomProjection(n_components='auto',
                                                      eps=0.9)

    def reset(self):
        self.embedding_list = []

    def reshape_embedding(self, embedding):
        embedding_list = []
        print("embedding shape: {}".format(embedding.shape))
        for k in range(embedding.shape[0]):
            for i in range(embedding.shape[2]):
                for j in range(embedding.shape[3]):
                    embedding_list.append(embedding[k, :, i, j])
        return embedding_list

    def add_embedding(self, prediction):
        temp_embedding = self.reshape_embedding(np.array(prediction))  # n*h*w, c
        self.embedding_list.extend(temp_embedding)

    def save_embedding(self):
        total_embeddings = np.array(self.embedding_list)
        self.randomprojector.fit(total_embeddings)
        # Coreset Subsampling
        selector = kCenterGreedy(total_embeddings, 0, 0)
        selected_idx = selector.select_batch(model=self.randomprojector, already_selected=[],
                                             N=int(total_embeddings.shape[0] * float(self.sampling_ratio)))
        embedding_coreset = total_embeddings[selected_idx]
        print('initial embedding size : ', total_embeddings.shape)
        print('final embedding size : ', embedding_coreset.shape)
        with open(self.save_path, 'wb') as f:
            pickle.dump(embedding_coreset, f)

    def __call__(self, prediction):
        embedding_coreset = pickle.load(open(self.save_path, 'rb'))
        if prediction.ndim == 3:
            prediction = np.expand_dims(prediction, 0)
        embedding_test = np.array(self.reshape_embedding(np.array(prediction)))
        if self.method == "KNN":
            knn = KNN(torch.from_numpy(embedding_coreset).cuda(), k=self.n_neighbors)
            score_patches = knn(torch.from_numpy(embedding_test).cuda())[0].cpu().detach().numpy()
        else:
            nbrs = NearestNeighbors(n_neighbors=self.neighbor_count,
                                    algorithm='ball_tree',
                                    metric='minkowski', p=2).fit(embedding_coreset)
            score_patches, _ = nbrs.kneighbors(embedding_test)
        N_b = score_patches[np.argmax(score_patches[:, 0])]
        w = (1 - (np.max(np.exp(N_b)) / np.sum(np.exp(N_b))))
        score = w * max(score_patches[:, 0])  # Image-level score
        if score > self.threshold:
            class_index = 1
        else:
            class_index = 0
        return class_index, score
