#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import numpy as np
from easy_pc.dataloader.utility import box3d_op


class Det3dPointCloudAugment():

    def __init__(self, shuffle_points=True):
        self.shuffle_points = shuffle_points
        self.flip_x = True
        self.flip_y = False
        self.global_transform = True

    def augment(self, src_cloud, gt_bboxes_3d):
        points = src_cloud[:]
        if self.flip_x or self.flip_y:
            gt_bboxes_3d, points = self.random_flip(gt_bboxes_3d,
                                                    points,
                                                    0.5,
                                                    self.flip_x,
                                                    self.flip_y)
        if self.global_transform:
            gt_bboxes_3d, points = self.global_rotation_v2(gt_bboxes_3d,
                                                           points)
            gt_bboxes_3d, points = self.global_scaling_v2(gt_bboxes_3d,
                                                          points)
        if self.shuffle_points:
            np.random.shuffle(points)
        return gt_bboxes_3d, points

    def noise_per_object(self, gt_boxes, points=None, valid_mask=None,
                         rotation_perturb=(-0.78539816, 0.78539816),
                         center_noise_std=(1.0, 1.0, 0.5),
                         global_random_rot_range=(0.0, 0.0),
                         num_try=100):
        """Random rotate or remove each groundtruth independently. use kitti viewer
        to test this function points_transform_

        Args:
            gt_boxes (np.ndarray): Ground truth boxes with shape (N, 7).
            points (np.ndarray, optional): Input point cloud with
                shape (M, 4). Default: None.
            valid_mask (np.ndarray, optional): Mask to indicate which
                boxes are valid. Default: None.
            rotation_perturb (float, optional): Rotation perturbation.
                Default: pi / 4.
            center_noise_std (float, optional): Center noise standard deviation.
                Default: 1.0.
            global_random_rot_range (float, optional): Global random rotation
                range. Default: pi/4.
            num_try (int, optional): Number of try. Default: 100.
        """
        num_boxes = gt_boxes.shape[0]
        if not isinstance(rotation_perturb, (list, tuple, np.ndarray)):
            rotation_perturb = [-rotation_perturb, rotation_perturb]
        if not isinstance(global_random_rot_range, (list, tuple, np.ndarray)):
            global_random_rot_range = [
                -global_random_rot_range, global_random_rot_range
            ]
        enable_grot = np.abs(global_random_rot_range[0] -
                             global_random_rot_range[1]) >= 1e-3

        if not isinstance(center_noise_std, (list, tuple, np.ndarray)):
            center_noise_std = [
                center_noise_std, center_noise_std, center_noise_std
            ]
        if valid_mask is None:
            valid_mask = np.ones((num_boxes,), dtype=np.bool_)
        center_noise_std = np.array(center_noise_std, dtype=gt_boxes.dtype)

        loc_noises = np.random.normal(
            scale=center_noise_std, size=[num_boxes, num_try, 3])
        rot_noises = np.random.uniform(
            rotation_perturb[0], rotation_perturb[1], size=[num_boxes, num_try])
        gt_grots = np.arctan2(gt_boxes[:, 0], gt_boxes[:, 1])
        grot_lowers = global_random_rot_range[0] - gt_grots
        grot_uppers = global_random_rot_range[1] - gt_grots
        global_rot_noises = np.random.uniform(
            grot_lowers[..., np.newaxis],
            grot_uppers[..., np.newaxis],
            size=[num_boxes, num_try])

        origin = (0.5, 0.5, 0)
        gt_box_corners = box3d_op.center_to_corner_box3d(
                    gt_boxes[:, :3],
                    gt_boxes[:, 3:6],
                    gt_boxes[:, 6],
                    origin=origin,
                    axis=2)

        # TODO: rewrite this noise box function?
        if not enable_grot:
            selected_noise = box3d_op.noise_per_box(gt_boxes[:, [0, 1, 3, 4, 6]],
                                                    valid_mask, loc_noises, rot_noises)
        else:
            selected_noise = box3d_op.noise_per_box_v2_(gt_boxes[:, [0, 1, 3, 4, 6]],
                                                        valid_mask, loc_noises, rot_noises,
                                                        global_rot_noises)

        loc_transforms = self._select_transform(loc_noises, selected_noise)
        rot_transforms = self._select_transform(rot_noises, selected_noise)
        surfaces = box3d_op.corner_to_surfaces_3d_jit(gt_box_corners)
        if points is not None:
            # TODO: replace this points_in_convex function by my tools?
            point_masks = box3d_op.points_in_convex_polygon_3d_jit(
                points[:, :3], surfaces)
            self.points_transform_(points, gt_boxes[:, :3],
                                   point_masks, loc_transforms,
                                   rot_transforms, valid_mask)

        self.box3d_transform_(gt_boxes, loc_transforms, rot_transforms, valid_mask)

    def random_flip(self, gt_boxes, points, probability=0.5,
                    random_flip_x=True, random_flip_y=False):
        flip_x = np.random.choice([False, True],
                                  replace=False,
                                  p=[1 - probability, probability])
        flip_y = np.random.choice([False, True],
                                  replace=False,
                                  p=[1 - probability, probability])
        if flip_y and random_flip_y:
            gt_boxes[:, 1] = -gt_boxes[:, 1]
            gt_boxes[:, 6] = -gt_boxes[:, 6] + np.pi
            points[:, 1] = -points[:, 1]
        if flip_x and random_flip_x:
            gt_boxes[:, 0] = -gt_boxes[:, 0]
            gt_boxes[:, 6] = -gt_boxes[:, 6]
            points[:, 0] = -points[:, 0]

        return gt_boxes, points

    def global_rotation_v2(self, gt_boxes, points,
                           min_rad=-np.pi / 4,
                           max_rad=np.pi / 4):
        noise_rotation = np.random.uniform(min_rad, max_rad)
        points[:, :3] = box3d_op.rotation_points_single_angle(
            points[:, :3], noise_rotation, axis=2)
        gt_boxes[:, :3] = box3d_op.rotation_points_single_angle(
            gt_boxes[:, :3], noise_rotation, axis=2)
        gt_boxes[:, 6] += noise_rotation
        return gt_boxes, points

    def global_scaling_v2(self, gt_boxes, points,
                          min_scale=0.95, max_scale=1.05):
        noise_scale = np.random.uniform(min_scale, max_scale)
        points[:, :3] *= noise_scale
        gt_boxes[:, :6] *= noise_scale
        if gt_boxes.shape[1] == 9:
            gt_boxes[:, 7:] *= noise_scale
        return gt_boxes, points

    def global_translate(self, gt_boxes, points,
                         noise_translate_std=(0, 0, 0)):
        """
        Apply global translation to gt_boxes and points.
        """

        if not isinstance(noise_translate_std, (list, tuple, np.ndarray)):
            noise_translate_std = np.array([noise_translate_std, noise_translate_std, noise_translate_std])
        if all([e == 0 for e in noise_translate_std]):
            return gt_boxes, points
        translation_std = np.array(noise_translate_std, dtype=np.float32)
        trans_factor = np.random.normal(scale=translation_std, size=3).T

        points[:, :3] += trans_factor
        gt_boxes[:, :3] += trans_factor

    def _select_transform(self, transform, indices):
        """Select transform.

        Args:
            transform (np.ndarray): Transforms to select from.
            indices (np.ndarray): Mask to indicate which transform to select.

        Returns:
            np.ndarray: Selected transforms.
        """
        result = np.zeros((transform.shape[0], *transform.shape[2:]),
                          dtype=transform.dtype)
        for i in range(transform.shape[0]):
            if indices[i] != -1:
                result[i] = transform[i, indices[i]]
        return result

    def points_transform_(self, points, centers, point_masks, loc_transform,
                          rot_transform, valid_mask):
        """Apply transforms to points and box centers.

        Args:
            points (np.ndarray): Input points.
            centers (np.ndarray): Input box centers.
            point_masks (np.ndarray): Mask to indicate which points need
                to be transformed.
            loc_transform (np.ndarray): Location transform to be applied.
            rot_transform (np.ndarray): Rotation transform to be applied.
            valid_mask (np.ndarray): Mask to indicate which boxes are valid.
        """
        num_box = centers.shape[0]
        num_points = points.shape[0]
        rot_mat_T = np.zeros((num_box, 3, 3), dtype=points.dtype)
        for i in range(num_box):
            self._rotation_matrix_3d_(rot_mat_T[i], rot_transform[i], 2)
        for i in range(num_points):
            for j in range(num_box):
                if valid_mask[j]:
                    if point_masks[i, j] == 1:
                        points[i, :3] -= centers[j, :3]
                        points[i:i + 1, :3] = points[i:i + 1, :3] @ rot_mat_T[j]
                        points[i, :3] += centers[j, :3]
                        points[i, :3] += loc_transform[j]
                        break  # only apply first box's transform

    def box3d_transform_(self, boxes, loc_transform, rot_transform, valid_mask):
        """Transform 3D boxes.

        Args:
            boxes (np.ndarray): 3D boxes to be transformed.
            loc_transform (np.ndarray): Location transform to be applied.
            rot_transform (np.ndarray): Rotation transform to be applied.
            valid_mask (np.ndarray): Mask to indicate which boxes are valid.
        """
        num_box = boxes.shape[0]
        for i in range(num_box):
            if valid_mask[i]:
                boxes[i, :3] += loc_transform[i]
                boxes[i, 6] += rot_transform[i]

    def _rotation_matrix_3d_(self, rot_mat_T, angle, axis):
        """Get the 3D rotation matrix.

        Args:
            rot_mat_T (np.ndarray): Transposed rotation matrix.
            angle (float): Rotation angle.
            axis (int): Rotation axis.
        """
        rot_sin = np.sin(angle)
        rot_cos = np.cos(angle)
        rot_mat_T[:] = np.eye(3)
        if axis == 1:
            rot_mat_T[0, 0] = rot_cos
            rot_mat_T[0, 2] = rot_sin
            rot_mat_T[2, 0] = -rot_sin
            rot_mat_T[2, 2] = rot_cos
        elif axis == 2 or axis == -1:
            rot_mat_T[0, 0] = rot_cos
            rot_mat_T[0, 1] = rot_sin
            rot_mat_T[1, 0] = -rot_sin
            rot_mat_T[1, 1] = rot_cos
        elif axis == 0:
            rot_mat_T[1, 1] = rot_cos
            rot_mat_T[1, 2] = rot_sin
            rot_mat_T[2, 1] = -rot_sin
            rot_mat_T[2, 2] = rot_cos
