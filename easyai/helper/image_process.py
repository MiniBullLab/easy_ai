#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
import numpy as np
from PIL import Image
import cv2


class ImageProcess():

    def __init__(self):
        pass

    def isImageFile(self, imagePath):
        if os.path.exists(imagePath):
            return any(imagePath.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"])
        else:
            return False

    def loadYUVImage(self, filepath):
        img = Image.open(filepath).convert('YCbCr')
        y, _, _ = img.split()
        return y

    def readRgbImage(self, imagePath):
        rgbImage = None
        srcImage = cv2.imread(imagePath)  # BGR
        if srcImage is not None:
            rgbImage = cv2.cvtColor(srcImage, cv2.COLOR_BGR2RGB)
        return srcImage, rgbImage

    def read_gray_image(self, image_path):
        gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        return gray_image

    def opencvImageRead(self, imagePath):
        img = cv2.imread(imagePath)  # BGR
        return img

    def opencv_save_image(self, image_path, image_data):
        cv2.imwrite(image_path, image_data)

    def cv2pil(self, image):
        """
        将bgr格式的numpy的图像转换为pil
        :param image:   图像数组
        :return:    Image对象
        """
        assert isinstance(image, np.ndarray), 'input image type is not cv2'
        if len(image.shape) == 2:
            return Image.fromarray(image)
        elif len(image.shape) == 3:
            return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    def pil2cv(self, image):
        """
        将Image对象转换为ndarray格式图像
        :param image:   图像对象
        :return:    ndarray图像数组
        """
        if len(image.split()) == 1:
            return np.asarray(image)
        elif len(image.split()) == 3:
            return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        elif len(image.split()) == 4:
            return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGBA2BGR)
