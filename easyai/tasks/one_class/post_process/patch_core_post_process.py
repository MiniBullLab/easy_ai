#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
import numpy as np
import cv2
import hnswlib
from sklearn.random_projection import SparseRandomProjection
from sklearn.neighbors import NearestNeighbors
from easyai.tasks.one_class.post_process.kcenter_greedy import kCenterGreedy
from easyai.tasks.utility.base_post_process import BasePostProcess
from easyai.name_manager.post_process_name import PostProcessName
from easyai.tasks.utility.task_registry import REGISTERED_POST_PROCESS
from easyai.utility.logger import EasyLogger


@REGISTERED_POST_PROCESS.register_module(PostProcessName.PatchCorePostProcess)
class PatchCorePostProcess(BasePostProcess):

    def __init__(self, save_path, threshold,
                 sampling_ratio=0.01, neighbor_count=9,
                 output_channel=1440, method="KNN"):
        super().__init__()
        self.save_path = save_path
        self.threshold = threshold
        self.sampling_ratio = sampling_ratio
        self.neighbor_count = neighbor_count
        self.output_channel = output_channel
        self.method = method
        self.embedding_list = []
        # Random projection
        # 'auto' => Johnson-Lindenstrauss lemma
        self.randomprojector = SparseRandomProjection(n_components='auto', eps=0.9)

    def reset(self):
        self.embedding_list = []

    def reshape_embedding(self, embedding):
        embedding_list = []
        EasyLogger.debug("embedding shape: {}".format(embedding.shape))
        for k in range(embedding.shape[0]):
            for i in range(embedding.shape[2]):
                for j in range(embedding.shape[3]):
                    embedding_list.append(embedding[k, :, i, j])
        return embedding_list

    def add_embedding(self, prediction):
        print("training: {}".format(prediction.shape))
        temp_embedding = self.reshape_embedding(np.array(prediction))  # n*h*w, c
        self.embedding_list.extend(temp_embedding)

    def save_embedding(self):
        total_embeddings = np.array(self.embedding_list)
        self.randomprojector.fit(total_embeddings)
        # Coreset Subsampling
        sampling_count = int(total_embeddings.shape[0] * float(self.sampling_ratio))
        selector = kCenterGreedy(total_embeddings, 0)
        selected_idx = selector.select_batch(model=self.randomprojector,
                                             already_selected=[],
                                             N=sampling_count)
        embedding_coreset = total_embeddings[selected_idx]
        EasyLogger.debug('initial embedding size : {}'.format(total_embeddings.shape))
        EasyLogger.debug('final embedding size : {}'.format(embedding_coreset.shape))
        embedding_coreset.tofile(self.save_path)

    def __call__(self, prediction):
        if not os.path.exists(self.save_path):
            EasyLogger.error("%s: embedding not exit!" % self.save_path)
            return None, 0
        embedding_data = np.fromfile(self.save_path, dtype=np.float32)
        embedding_coreset = embedding_data.reshape(-1, self.output_channel)
        if prediction.ndim == 3:
            prediction = np.expand_dims(prediction, axis=0)
        embedding_test = np.array(self.reshape_embedding(np.array(prediction)),
                                  dtype=np.float32)
        if self.method == "KNN":
            knn = cv2.ml.KNearest_create()
            knn.setAlgorithmType(cv2.ml.KNEAREST_BRUTE_FORCE)
            labels = [0 for _ in range(embedding_coreset.shape[0])]
            labels = np.asarray(labels)
            knn.train(embedding_coreset, cv2.ml.ROW_SAMPLE, labels)
            result = knn.findNearest(embedding_test, k=self.neighbor_count)
            score_patches = result[-1]
        elif self.method == "ANN":
            # Declaring index，声明索引类型，如：l2, cosine or ip
            nbrs = hnswlib.Index(space='l2', dim=len(embedding_coreset[0]))
            # 初始化索引，元素的最大数需要是已知的
            nbrs.init_index(max_elements=len(embedding_coreset), ef_construction=self.neighbor_count * 10, M=16)
            # Element insertion，插入数据
            int_labels = nbrs.add_items(embedding_coreset, np.arange(len(embedding_coreset)))
            nbrs.set_ef(self.neighbor_count * 10)
            _, score_patches = nbrs.knn_query(embedding_test, self.neighbor_count)
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
