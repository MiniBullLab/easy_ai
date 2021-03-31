#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import numpy as np
import torchvision.utils as vutils


class TorchVisionVisualizer():

    def __init__(self):
        pass

    @staticmethod
    def normalize(inp):
        """Normalize the tensor

        Args:
            inp ([FloatTensor]): Input tensor

        Returns:
            [FloatTensor]: Normalized tensor.
        """
        return (inp - inp.min()) / (inp.max() - inp.min() + 1e-5)

    def display_current_images(self, fakes):
        """ Display current images.

        Args:
            fakes ([FloatTensor]): Fake Image
        """
        fakes = self.normalize(fakes.cpu().numpy())
        self.vis.images(fakes, win=2, opts={'title': 'Fakes'})

    def save_current_images(self, fakes, save_path):
        """ Save images.
        Args:
            fakes ([FloatTensor]): Fake Image
        """
        vutils.save_image(fakes, save_path, normalize=True)
