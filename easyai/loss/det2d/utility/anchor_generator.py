#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import math
import torch
import numpy as np
from torch import nn


class BufferList(nn.Module):
    """
    Similar to nn.ParameterList, but for buffers
    """

    def __init__(self, buffers=None):
        super(BufferList, self).__init__()
        if buffers is not None:
            self.extend(buffers)

    def extend(self, buffers):
        offset = len(self)
        for i, buffer in enumerate(buffers):
            self.register_buffer(str(offset + i), buffer)
        return self

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())


class SSDPriorBoxGenerator():

    def __init__(self, anchor_counts, aspect_ratios,
                 min_sizes, max_sizes):
        self.anchor_counts = anchor_counts
        self.aspect_ratios = aspect_ratios
        self.min_sizes = min_sizes
        self.max_sizes = max_sizes
        self.clip = False

    def __call__(self, input_size, feature_sizes):
        image_w, image_h = input_size
        anchor_boxes_list = []
        for index, feature_size in enumerate(feature_sizes):
            feature_map_w, feature_map_h = feature_size
            stride_w = image_w / feature_map_w
            stride_h = image_h / feature_map_h

            boxes = []
            stride_offset_w, stride_offset_h = 0.5 * stride_w, 0.5 * stride_h
            boxes.append((stride_offset_w, stride_offset_h,
                          self.min_sizes[index], self.min_sizes[index]))
            extra_s = math.sqrt(self.min_sizes[index] * self.max_sizes[index])
            boxes.append((stride_offset_w, stride_offset_h, extra_s, extra_s))

            for ratio in self.aspect_ratios[index]:
                boxes.append((stride_offset_w, stride_offset_h,
                              self.min_sizes[index] * math.sqrt(ratio),
                              self.min_sizes[index] / math.sqrt(ratio)))
                boxes.append((stride_offset_w, stride_offset_h,
                              self.min_sizes[index] / math.sqrt(ratio),
                              self.min_sizes[index] * math.sqrt(ratio)))

            anchor_bases = torch.FloatTensor(np.array(boxes))
            assert anchor_bases.size(0) == self.anchor_counts[index]
            anchors = anchor_bases.contiguous().view(1, -1, 4).\
                repeat(feature_map_h * feature_map_w, 1, 1).contiguous().view(-1, 4)
            grid_len_h = np.arange(0, image_h - stride_offset_h, stride_h)
            grid_len_w = np.arange(0, image_w - stride_offset_w, stride_w)
            a, b = np.meshgrid(grid_len_w, grid_len_h)

            x_offset = torch.FloatTensor(a).view(-1, 1)
            y_offset = torch.FloatTensor(b).view(-1, 1)

            x_y_offset = torch.cat((x_offset, y_offset), 1).contiguous().view(-1, 1, 2)
            x_y_offset = x_y_offset.repeat(1, self.anchor_counts[index], 1).contiguous().view(-1, 2)
            anchors[:, :2] = anchors[:, :2] + x_y_offset

            if self.clip:
                anchors[:, 0::2].clamp_(min=0., max=image_w - 1)
                anchors[:, 1::2].clamp_(min=0., max=image_h - 1)
            anchor_boxes_list.append(anchors)
        return torch.cat(anchor_boxes_list, 0)


class PriorBoxGenerator(nn.Module):

    def __init__(self, input_size=(320, 256),
                 min_sizes=([10, 16, 32], 64, 128, 256),
                 max_sizes=([10, 16, 32], 64, 128, 256),
                 aspect_ratios=([2.0, 0.5], [2.0, 0.5], [2.0, 0.5], [2.0, 0.5]),
                 anchor_strides=([8, 8], [16, 16], [32, 32], [64, 64])):
        super().__init__()
        self.input_size = input_size
        self.min_sizes = min_sizes
        self.max_sizes = max_sizes
        self.aspect_ratios = aspect_ratios
        self.anchor_strides = anchor_strides
        self.use_max_sizes = False
        self.clip = True

    def point_form(self, boxes):
        """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
        representation for comparison to point form ground truth data.
        Args:
            boxes: (tensor) center-size default boxes from priorbox layers.
        Return:
            boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
        """
        return torch.cat(
            (
                boxes[:, :2] - boxes[:, 2:] / 2,  # xmin, ymin
                boxes[:, :2] + boxes[:, 2:] / 2),
            1)  # xmax, ymax

    def center_size(self, boxes):
        """ Convert prior_boxes to (cx, cy, w, h)
        representation for comparison to center-size form ground truth data.
        Args:
            boxes: (tensor) point_form boxes
        Return:
            boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
        """
        return torch.cat(
            [(boxes[:, 2:] + boxes[:, :2]) / 2, boxes[:, 2:] - boxes[:, :2]],
            1)  # w, h

    def forward(self, input_size, feature_sizes):
        self.input_size = input_size
        mean = []
        for k, feature in enumerate(self.feature_sizes):
            grid_h, grid_w = feature[1], feature[0]
            for i in range(grid_h):
                for j in range(grid_w):
                    if isinstance(self.min_sizes[k], int):
                        f_k_h = self.input_size[1] / self.anchor_strides[k][1]
                        f_k_w = self.input_size[0] / self.anchor_strides[k][0]
                        # unit center x,y
                        cx = (j + 0.5) / f_k_w
                        cy = (i + 0.5) / f_k_h

                        # aspect_ratio: 1
                        # rel size: min_size
                        s_k_h = self.min_sizes[k] / self.input_size[1]
                        s_k_w = self.min_sizes[k] / self.input_size[0]
                        mean += [cx, cy, s_k_w, s_k_h]

                        if self.use_max_sizes:
                            s_k_prime_w = math.sqrt(s_k_w * (self.max_sizes[k] / self.input_size[0]))
                            s_k_prime_h = math.sqrt(s_k_h * (self.max_sizes[k] / self.input_size[1]))
                            mean += [cx, cy, s_k_prime_w, s_k_prime_h]

                        for ar in self.aspect_ratios[k]:
                            mean += [cx, cy, s_k_w * math.sqrt(ar), s_k_h / math.sqrt(ar)]
                    elif isinstance(self.min_sizes[k], list):
                        if self.use_max_sizes:
                            for min_sizes, max_sizes in zip(self.min_sizes[k], self.max_sizes[k]):
                                if len(self.min_sizes[k]) != len(self.max_sizes[k]):
                                    raise Exception('the max size must have same formate')
                                f_k_h = self.input_size[1] / self.anchor_strides[k][1]
                                f_k_w = self.input_size[0] / self.anchor_strides[k][0]
                                # unit center x,y
                                cx = (j + 0.5) / f_k_w
                                cy = (i + 0.5) / f_k_h

                                # aspect_ratio: 1
                                # rel size: min_size
                                s_k_h = min_sizes / self.input_size[1]
                                s_k_w = min_sizes / self.input_size[0]
                                mean += [cx, cy, s_k_w, s_k_h]

                                # aspect_ratio: 1
                                # rel size: sqrt(s_k * s_(k+1))

                                s_k_prime_w = math.sqrt(
                                    s_k_w * (max_sizes / self.img_wh[0]))
                                s_k_prime_h = math.sqrt(
                                    s_k_h * (max_sizes / self.img_wh[1]))
                                mean += [cx, cy, s_k_prime_w, s_k_prime_h]

                                for ar in self.aspect_ratios[k]:
                                    mean += [cx, cy, s_k_w * math.sqrt(ar), s_k_h / math.sqrt(ar)]
                        else:
                            for min_sizes in self.min_sizes[k]:
                                f_k_h = self.input_size[1] / self.anchor_strides[k][1]
                                f_k_w = self.input_size[0] / self.anchor_strides[k][0]
                                # unit center x,y
                                cx = (j + 0.5) / f_k_w
                                cy = (i + 0.5) / f_k_h

                                # aspect_ratio: 1
                                # rel size: min_size
                                s_k_h = min_sizes / self.img_wh[1]
                                s_k_w = min_sizes / self.img_wh[0]
                                mean += [cx, cy, s_k_w, s_k_h]
                                for ar in self.aspect_ratios[k]:
                                    mean += [cx, cy, s_k_w * math.sqrt(ar), s_k_h / math.sqrt(ar)]
                    else:
                        raise Exception('please check min_sizes')

        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output_point = self.point_form(output)
            output_point.clamp_(max=1, min=0)
            output = self.center_size(output_point)
        return output


class AnchorGenerator(nn.Module):
    """
    For a set of image sizes and feature maps, computes a set
    of anchors
    """

    def __init__(self, image_size,
                 sizes=(32, 64, 128, 256, 512),
                 aspect_ratios=(0.5, 1.0, 2.0),
                 anchor_strides=(4, 8, 16, 32, 64),
                 straddle_thresh=0):
        super().__init__()

        if len(anchor_strides) == 1:
            anchor_stride = anchor_strides[0]
            cell_anchors = [
                self.generate_anchors(anchor_stride, sizes, aspect_ratios).float()
            ]
        else:
            if len(anchor_strides) != len(sizes):
                raise RuntimeError("FPN should have #anchor_strides == #sizes")

            cell_anchors = [
                self.generate_anchors(
                    anchor_stride,
                    size if isinstance(size, (tuple, list)) else (size,),
                    aspect_ratios
                ).float()
                for anchor_stride, size in zip(anchor_strides, sizes)
            ]
        self.image_size = image_size
        self.strides = anchor_strides
        self.cell_anchors = BufferList(cell_anchors)
        self.straddle_thresh = straddle_thresh

    def num_anchors_per_location(self):
        return [len(cell_anchors) for cell_anchors in self.cell_anchors]

    def grid_anchors(self, grid_sizes):
        anchors = []
        for size, stride, base_anchors in zip(
            grid_sizes, self.strides, self.cell_anchors
        ):
            grid_height, grid_width = size
            device = base_anchors.device
            shifts_x = torch.arange(
                0, grid_width * stride, step=stride, dtype=torch.float32, device=device
            )
            shifts_y = torch.arange(
                0, grid_height * stride, step=stride, dtype=torch.float32, device=device
            )
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            anchors.append(
                (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4)
            )

        return anchors

    def get_visibility(self, bbox):
        image_width, image_height = self.image_size
        if self.straddle_thresh >= 0:
            inds_inside = (
                (bbox[..., 0] >= -self.straddle_thresh)
                & (bbox[..., 1] >= -self.straddle_thresh)
                & (bbox[..., 2] < image_width + self.straddle_thresh)
                & (bbox[..., 3] < image_height + self.straddle_thresh)
            )
        else:
            device = bbox.device
            inds_inside = torch.ones(bbox.shape[0], dtype=torch.bool, device=device)
        return inds_inside

    def set_image_size(self, size):
        self.image_size = size

    def forward(self, feature_maps):
        batch_size, _, _, _ = feature_maps[0].size()
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes)
        anchors = []
        inside_anchors = []
        for _ in range(batch_size):
            anchors_in_image = []
            index_inside = []
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                index_inside.append(self.get_visibility(anchors_per_feature_map))
                anchors_in_image.append(anchors_per_feature_map)
            inside_anchors.append(index_inside)
            anchors.append(anchors_in_image)
        return anchors, inside_anchors

    def generate_anchors(self, stride, sizes, aspect_ratios):
        """Generate anchor (reference) windows by enumerating aspect ratios X
        scales wrt a reference (0, 0, base_size - 1, base_size - 1) window.
        """
        base_size = stride
        scales = np.array(sizes, dtype=np.float) / stride
        aspect_ratios = np.array(aspect_ratios, dtype=np.float)
        anchor = np.array([1, 1, base_size, base_size], dtype=np.float) - 1
        anchors = self._ratio_enum(anchor, aspect_ratios)  # 3 * 4
        anchors = np.vstack(
            [self._scale_enum(anchors[i, :], scales) for i in range(anchors.shape[0])]
        )  # 15 * 4
        return torch.from_numpy(anchors)

    def _whctrs(self, anchor):
        """Return width, height, x center, and y center for an anchor (window)."""
        w = anchor[2] - anchor[0] + 1
        h = anchor[3] - anchor[1] + 1
        x_ctr = anchor[0] + 0.5 * (w - 1)
        y_ctr = anchor[1] + 0.5 * (h - 1)
        return w, h, x_ctr, y_ctr

    def _mkanchors(self, ws, hs, x_ctr, y_ctr):
        """Given a vector of widths (ws) and heights (hs) around a center
        (x_ctr, y_ctr), output a set of anchors (windows).
        """
        ws = ws[:, np.newaxis]
        hs = hs[:, np.newaxis]
        anchors = np.hstack(
            (
                x_ctr - 0.5 * (ws - 1),
                y_ctr - 0.5 * (hs - 1),
                x_ctr + 0.5 * (ws - 1),
                y_ctr + 0.5 * (hs - 1),
            )
        )
        return anchors

    def _ratio_enum(self, anchor, ratios):
        """Enumerate a set of anchors for each aspect ratio wrt an anchor."""
        w, h, x_ctr, y_ctr = self._whctrs(anchor)
        size = w * h
        size_ratios = size / ratios
        ws = np.round(np.sqrt(size_ratios))
        hs = np.round(ws * ratios)
        anchors = self._mkanchors(ws, hs, x_ctr, y_ctr)
        return anchors

    def _scale_enum(self, anchor, scales):
        """Enumerate a set of anchors for each scale wrt an anchor."""
        w, h, x_ctr, y_ctr = self._whctrs(anchor)
        ws = w * scales
        hs = h * scales
        anchors = self._mkanchors(ws, hs, x_ctr, y_ctr)
        return anchors
