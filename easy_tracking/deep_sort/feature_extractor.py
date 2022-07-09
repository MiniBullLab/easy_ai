import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging
from easyai.torch_utility.torch_model_process import TorchModelProcess


class Extractor(object):
    def __init__(self, model_name, weights_path):
        self.model_process = TorchModelProcess()
        self.model_args = {"type": model_name,
                           "data_channel": 3,
                           "class_number": 751,
                           "reid": 512}
        self.model = self.model_process.create_model(self.model_args, 0)
        self.device = self.model_process.get_device()
        self.model_process.load_latest_model(weights_path, self.model)
        self.model = self.model_process.model_test_init(self.model)
        self.model.eval()
        self.size = (64, 128)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """
        def _resize(im, size):
            return cv2.resize(im, size)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(
            0) for im in im_crops], dim=0).float()
        return im_batch

    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.model(im_batch)[0]
        return features.cpu().numpy()


if __name__ == '__main__':
    img = cv2.imread("demo.jpg")[:, :, (2, 1, 0)]
    extr = Extractor("checkpoint/ckpt.t7")
    feature = extr(img)
    print(feature.shape)
