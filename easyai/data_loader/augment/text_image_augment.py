#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie
"""
@inproceedings{luo2020learn,
  author = {Canjie Luo and Yuanzhi Zhu and Lianwen Jin and Yongpan Wang},
  title = {Learn to Augment: Joint Data Augmentation and Network Optimization for Text Recognition},
  booktitle = {CVPR},
  year = {2020}
}
"""

import numpy as np
from easyai.data_loader.augment.warp_mls import WarpMLS


class TextImageAugment():

    def __init__(self):
        pass

    def distort(self, src_image, cut_count):
        img_h, img_w = src_image.shape[:2]

        cut = img_w // cut_count
        thresh = cut // 3
        # thresh = img_h // segment // 3
        # thresh = img_h // 5

        src_pts = list()
        dst_pts = list()

        src_pts.append([0, 0])
        src_pts.append([img_w, 0])
        src_pts.append([img_w, img_h])
        src_pts.append([0, img_h])

        dst_pts.append([np.random.randint(thresh), np.random.randint(thresh)])
        dst_pts.append([img_w - np.random.randint(thresh), np.random.randint(thresh)])
        dst_pts.append([img_w - np.random.randint(thresh), img_h - np.random.randint(thresh)])
        dst_pts.append([np.random.randint(thresh), img_h - np.random.randint(thresh)])

        half_thresh = thresh * 0.5

        for cut_idx in np.arange(1, cut_count, 1):
            src_pts.append([cut * cut_idx, 0])
            src_pts.append([cut * cut_idx, img_h])
            dst_pts.append([cut * cut_idx + np.random.randint(thresh) - half_thresh,
                            np.random.randint(thresh) - half_thresh])
            dst_pts.append([cut * cut_idx + np.random.randint(thresh) - half_thresh,
                            img_h + np.random.randint(thresh) - half_thresh])

        trans = WarpMLS(src_image, src_pts, dst_pts, img_w, img_h)
        dst = trans.generate()

        return dst

    def stretch(self, src_image, cut_count):
        img_h, img_w = src_image.shape[:2]

        cut = img_w // cut_count
        thresh = cut * 4 // 5
        # thresh = img_h // segment // 3
        # thresh = img_h // 5

        src_pts = list()
        dst_pts = list()

        src_pts.append([0, 0])
        src_pts.append([img_w, 0])
        src_pts.append([img_w, img_h])
        src_pts.append([0, img_h])

        dst_pts.append([0, 0])
        dst_pts.append([img_w, 0])
        dst_pts.append([img_w, img_h])
        dst_pts.append([0, img_h])

        half_thresh = thresh * 0.5

        for cut_idx in np.arange(1, cut_count, 1):
            move = np.random.randint(thresh) - half_thresh
            src_pts.append([cut * cut_idx, 0])
            src_pts.append([cut * cut_idx, img_h])
            dst_pts.append([cut * cut_idx + move, 0])
            dst_pts.append([cut * cut_idx + move, img_h])

        trans = WarpMLS(src_image, src_pts, dst_pts, img_w, img_h)
        dst = trans.generate()

        return dst

    def perspective(self, src_image):
        img_h, img_w = src_image.shape[:2]

        thresh = img_h // 2

        src_pts = list()
        dst_pts = list()

        src_pts.append([0, 0])
        src_pts.append([img_w, 0])
        src_pts.append([img_w, img_h])
        src_pts.append([0, img_h])

        dst_pts.append([0, np.random.randint(thresh)])
        dst_pts.append([img_w, np.random.randint(thresh)])
        dst_pts.append([img_w, img_h - np.random.randint(thresh)])
        dst_pts.append([0, img_h - np.random.randint(thresh)])

        trans = WarpMLS(src_image, src_pts, dst_pts, img_w, img_h)
        dst = trans.generate()

        return dst