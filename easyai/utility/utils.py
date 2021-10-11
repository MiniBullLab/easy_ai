import torch.nn as nn
import hashlib
import logging
import matplotlib.pyplot as plt


def calculate_md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


class vis_bn():
    """Summarize the given input model.
    Summarized information are 1) output shape, 2) kernel shape,
    3) number of the parameters and 4) operations (Mult-Adds)
    Args:
        model (Module): Model to summarize
        x (Tensor): Input tensor of the model with [N, C, H, W] shape
                    dtype and device have to match to the model
        args, kwargs: Other argument used in `model.forward` function

    descript:
    # 1. visual params from bn layer
    # 2. visual featuremap from con layer
    # 3. visual low dim of the last layer
    """

    def __init__(self, model):
        self.axss = []
        self.bn_layers = []
        self.bn_layer_name = []
        bn_layer_index = 0

        for name, p in model.named_parameters():
            if 'bn' in name and 'weight' in name:
                self.bn_layer_name.append('.'.join(name.split(".")[:-1]))

        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                f, axs = plt.subplots(4, 1, figsize=(6, 6))
                f.suptitle(self.bn_layer_name[bn_layer_index])
                bn_layer_index += 1
                self.axss.append(axs)
                self.bn_layers.append(m)

    def plot(self):
        for i, axs in enumerate(self.axss):
            m = self.bn_layers[i]
            self.plot_hist(axs, m.weight.data, m.bias.data, m.running_mean.data, m.running_var.data)

    def plot_hist(self, axs, weight, bias, running_mean, running_var):
        [a.clear() for a in [axs[0], axs[1], axs[2], axs[3]]]
        axs[0].bar(range(len(running_mean.cpu().numpy())), weight.cpu().numpy(), color='#FF9359')
        axs[1].bar(range(len(running_var.cpu().numpy())), bias.cpu().numpy(), color='g')
        axs[2].bar(range(len(running_mean.cpu().numpy())), running_mean.cpu().numpy(), color='#74BCFF')
        axs[3].bar(range(len(running_var.cpu().numpy())), running_var.cpu().numpy(), color='y')
        axs[0].set_ylabel('weight')
        axs[1].set_ylabel('bias')
        axs[2].set_ylabel('running_mean')
        axs[3].set_ylabel('running_var')
        plt.pause(0.01)