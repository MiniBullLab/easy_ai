"""Returns points that minimizes the maximum distance of any point to a center.

Implements the k-Center-Greedy method in
Ozan Sener and Silvio Savarese.  A Geometric Approach to Active Learning for
Convolutional Neural Networks. https://arxiv.org/abs/1708.00489 2017

Distance metric defaults to l2 distance.  Features used to calculate distance
are either raw features or if a model has transform method then uses the output
of model.transform(X).

Can be extended to a robust k centers algorithm that ignores a certain number of
outlier datapoints.  Resulting centers are solution to multiple integer program.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from sklearn.metrics import pairwise_distances
from easyai.utility.logger import EasyLogger


class kCenterGreedy():

    def __init__(self, X, y, metric='euclidean'):
        self.X = X
        self.y = y
        self.flat_X = self.flatten_X()
        self.features = self.flat_X
        self.metric = metric
        self.min_distances = None
        self.n_obs = self.X.shape[0]
        self.already_selected = []

    def flatten_X(self):
        shape = self.X.shape
        flat_X = self.X
        if len(shape) > 2:
            flat_X = np.reshape(self.X, (shape[0], np.product(shape[1:])))
        return flat_X

    def update_distances(self, cluster_centers, only_new=True, reset_dist=False):
        """
        Update min distances given cluster centers.

        Args:
          reset_dist: whether to reset min_distances.
          cluster_centers: indices of cluster centers
          only_new: only calculate distance for newly selected points
          and update min_distances.
        """
        if reset_dist:
            self.min_distances = None
        if only_new:
            cluster_centers = [d for d in cluster_centers
                               if d not in self.already_selected]
        if cluster_centers:
            # Update min_distances for all examples given new cluster center.
            x = self.features[cluster_centers]
            dist = pairwise_distances(self.features, x, metric=self.metric)

            if self.min_distances is None:
                self.min_distances = np.min(dist, axis=1).reshape(-1, 1)
            else:
                self.min_distances = np.minimum(self.min_distances, dist)

    def select_batch(self, model, already_selected, N):
        """
        Diversity promoting active learning method that greedily forms a batch
        to minimize the maximum distance to a cluster center among all unlabeled
        datapoints.

        Args:
          model: model with scikit-like API with decision_function implemented
          already_selected: index of datapoints already selected
          N: batch size 选择点的个数

        Returns:
          indices of points selected to minimize distance to cluster centers
        """

        try:
            # Assumes that the transform function takes in original data and not
            # flattened data.
            EasyLogger.debug('Getting transformed features...')
            self.features = model.transform(self.X)
            EasyLogger.debug("Sparse shape: {}".format(self.features.shape))
            self.update_distances(already_selected, only_new=False, reset_dist=True)
        except Exception as err:
            EasyLogger.debug('Using flat_X as features.')
            self.update_distances(already_selected, only_new=True, reset_dist=False)

        new_batch = []
        for _ in range(N):
            if self.already_selected is None:
                # Initialize centers with a randomly selected datapoint 初始化一个中心点
                ind = np.random.choice(np.arange(self.n_obs))
            else:
                ind = np.argmax(self.min_distances)  # 选择距离中最大的下标点
            # New examples should not be in already selected since those points
            # should have min_distance of zero to a cluster center.
            assert ind not in already_selected

            # 更新初始点到其余点的距离
            self.update_distances([ind], only_new=True, reset_dist=False)
            new_batch.append(ind)
        EasyLogger.debug('Maximum distance from cluster centers is %0.2f' % max(self.min_distances))

        self.already_selected = already_selected

        return new_batch
