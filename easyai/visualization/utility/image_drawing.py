#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
import inspect
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from easyai.visualization.utility.color_define import ColorDefine


class ImageDrawing():

    def __init__(self):
        pass

    def draw_image(self, widnow_name, image, scale=1.0):
        cv2.namedWindow(widnow_name, 0)
        cv2.resizeWindow(widnow_name, int(image.shape[1] * scale), int(image.shape[0] * scale))
        cv2.imshow(widnow_name, image)

    def draw_detect_objects(self, src_image, result_objects):
        for object in result_objects:
            point1 = (int(object.min_corner.x), int(object.min_corner.y))
            point2 = (int(object.max_corner.x), int(object.max_corner.y))
            index = object.classIndex
            cv2.rectangle(src_image, point1, point2, ColorDefine.colors[index], 2)

    def draw_segment_result(self, src_image, result, class_list):
        r = result.copy()
        g = result.copy()
        b = result.copy()
        for index, value_data in enumerate(class_list):
            value = value_data[1]
            color_list = [int(x) for x in value.split(',') if x.strip()]
            if len(color_list) == 1:
                r[result == index] = color_list[0]
                g[result == index] = color_list[0]
                b[result == index] = color_list[0]
            elif len(color_list) == 3:
                r[result == index] = color_list[0]
                g[result == index] = color_list[1]
                b[result == index] = color_list[2]
            else:
                r[result == index] = 0
                g[result == index] = 0
                b[result == index] = 0

        rgb = np.zeros((result.shape[0], result.shape[1], 3))

        rgb[:, :, 0] = (r * 1.0 + src_image[:, :, 2] * 0) / 255.0
        rgb[:, :, 1] = (g * 1.0 + src_image[:, :, 1] * 0) / 255.0
        rgb[:, :, 2] = (b * 1.0 + src_image[:, :, 0] * 0) / 255.0

        return rgb

    def draw_keypoint2d_result(self, src_image, result, skeleton):
        for result_object in result:
            key_points = result_object.get_key_points()
            for edge in skeleton:
                point1 = key_points[edge[0]]
                point2 = key_points[edge[1]]
                x1 = int(point1.x)
                y1 = int(point1.y)
                x2 = int(point2.x)
                y2 = int(point2.y)
                if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
                    continue
                cv2.line(src_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            for point in key_points:
                cv2.circle(src_image, (int(point.x), int(point.y)), 5, (0, 255, 225), 2)

    def draw_det_keypoint2d_result(self, src_image, result, skeleton):
        for result_object in result:
            point1 = (int(result_object.min_corner.x), int(result_object.min_corner.y))
            point2 = (int(result_object.max_corner.x), int(result_object.max_corner.y))
            cv2.rectangle(src_image, point1, point2, (255, 255, 255), 2)
            key_points = result_object.get_key_points()
            for edge in skeleton:
                point1 = key_points[edge[0]]
                point2 = key_points[edge[1]]
                x1 = int(point1.x)
                y1 = int(point1.y)
                x2 = int(point2.x)
                y2 = int(point2.y)
                if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
                    continue
                cv2.line(src_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            for point in key_points:
                cv2.circle(src_image, (int(point.x), int(point.y)), 5, (0, 255, 225), 2)

    def draw_polygon2d_result(self, src_image, result):
        for result_object in result:
            polygon = result_object.get_polygon()
            point_list = []
            for point in polygon:
                x = int(point.x)
                y = int(point.y)
                point_list.append([x, y])
            cv2.polylines(src_image, np.array([point_list], np.int32),
                          True, (0, 0, 225), 3, 0)

    def draw_ocr_result(self, src_image, result):
        random.seed(0)
        current_path = inspect.getfile(inspect.currentframe())
        dir_name = os.path.join(os.path.dirname(current_path), "../../config/fonts")
        font_file = os.path.join(dir_name, "方正隶书简体.ttf")
        if isinstance(src_image, np.ndarray):
            src_image = Image.fromarray(src_image)
        h, w = src_image.height, src_image.width
        img_left = src_image.copy()
        img_right = Image.new('RGB', (w, h), (255, 255, 255))
        draw_left = ImageDraw.Draw(img_left)
        draw_right = ImageDraw.Draw(img_right)
        for temp_object in result:
            color = (random.randint(0, 255), random.randint(0, 255),
                     random.randint(0, 255))
            polygon = temp_object.get_polygon()
            draw_points = [(p.x, p.y) for p in polygon]
            draw_left.polygon(draw_points, fill=color)
            txt = temp_object.get_text()
            if temp_object.get_text() is not None:
                font = ImageFont.truetype(font_file, 18, encoding="utf-8")
                draw_right.text([draw_points[0][0], draw_points[0][1]],
                                txt, fill=(0, 0, 0), font=font)
        img_left = Image.blend(src_image, img_left, 0.5)
        img_show = Image.new('RGB', (w * 2, h), (255, 255, 255))
        img_show.paste(img_left, (0, 0, w, h))
        img_show.paste(img_right, (w, 0, w * 2, h))
        return np.array(img_show)

    def save_image(self, image, save_path):
        cv2.imwrite(save_path, image)

