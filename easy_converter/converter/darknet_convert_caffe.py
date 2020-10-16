#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import pathlib
import caffe
import numpy as np
from easy_converter.converter.darknet2caffe.cfg import *
from easy_converter.converter.darknet2caffe.prototxt import *
from optparse import OptionParser


def parse_arguments():
    parser = OptionParser()
    parser.description = "This program is keras convert to onnx"

    parser.add_option("-c", "--cfg", dest="cfg_path",
                      metavar="PATH", type="string", default=None,
                      help="darknet cfg path")

    parser.add_option("-w", "--weight", dest="weight_path",
                      type="string", default=None,
                      help="darknet weight path")

    (options, args) = parser.parse_args()

    return options


class DarknetConvertCaffe():

    def __init__(self, cfg_path, weight_path, header_count=4):
        self.cfg_path = pathlib.Path(cfg_path)
        self.weight_path = pathlib.Path(weight_path)
        self.header_count = header_count  # yolov2 is 4 or yolov3 is 5
        self.proto_path = self.cfg_path.with_suffix(".prototxt")
        self.caffe_model_save_path = self.cfg_path.with_suffix(".caffemodel")

    def convert_caffe(self):
        cfgfile = str(self.cfg_path)
        weightfile = str(self.weight_path)
        protofile = str(self.proto_path)
        caffemodel = str(self.caffe_model_save_path)

        net_info = self.cfg2prototxt(cfgfile)
        save_prototxt(net_info, protofile, region=False)

        net = caffe.Net(protofile, caffe.TEST)
        params = net.params

        blocks = parse_cfg(cfgfile)
        fp = open(weightfile, 'rb')
        header = np.fromfile(fp, count=self.header_count, dtype=np.int32)
        buf = np.fromfile(fp, dtype=np.float32)
        fp.close()

        layers = []
        layer_id = 0
        start = 0
        for block in blocks:
            if start >= buf.size:
                break

            if block['type'] == 'net':
                continue
            elif block['type'] == 'convolutional':
                batch_normalize = int(block['batch_normalize'])
                if 'name' in block:
                    conv_layer_name = block['name']
                    bn_layer_name = '%s-bn' % block['name']
                    scale_layer_name = '%s-scale' % block['name']
                else:
                    conv_layer_name = 'layer%d-conv' % layer_id
                    bn_layer_name = 'layer%d-bn' % layer_id
                    scale_layer_name = 'layer%d-scale' % layer_id

                if batch_normalize:
                    start = self.load_conv_bn2caffe(buf, start, params[conv_layer_name],
                                                    params[bn_layer_name],
                                                    params[scale_layer_name])
                else:
                    start = self.load_conv2caffe(buf, start, params[conv_layer_name])
                layer_id = layer_id + 1
            elif block['type'] == 'depthwise_convolutional':
                batch_normalize = int(block['batch_normalize'])
                if 'name' in block:
                    conv_layer_name = block['name']
                    bn_layer_name = '%s-bn' % block['name']
                    scale_layer_name = '%s-scale' % block['name']
                else:
                    conv_layer_name = 'layer%d-dwconv' % layer_id
                    bn_layer_name = 'layer%d-bn' % layer_id
                    scale_layer_name = 'layer%d-scale' % layer_id

                if batch_normalize:
                    start = self.load_conv_bn2caffe(buf, start, params[conv_layer_name],
                                                    params[bn_layer_name],
                                                    params[scale_layer_name])
                else:
                    start = self.load_conv2caffe(buf, start, params[conv_layer_name])
                layer_id = layer_id + 1
            elif block['type'] == 'connected':
                if 'name' in block:
                    fc_layer_name = block['name']
                else:
                    fc_layer_name = 'layer%d-fc' % layer_id
                start = self.load_fc2caffe(buf, start, params[fc_layer_name])
                layer_id = layer_id + 1
            elif block['type'] == 'maxpool':
                layer_id = layer_id + 1
            elif block['type'] == 'avgpool':
                layer_id = layer_id + 1
            elif block['type'] == 'reorg':
                layer_id = layer_id + 1
            elif block['type'] == 'region':
                layer_id = layer_id + 1
            elif block['type'] == 'yolo':
                layer_id = layer_id + 1
            elif block['type'] == 'route':
                layer_id = layer_id + 1
            elif block['type'] == 'shortcut':
                layer_id = layer_id + 1
            elif block['type'] == 'softmax':
                layer_id = layer_id + 1
            elif block['type'] == 'cost':
                layer_id = layer_id + 1
            elif block['type'] == 'upsample':
                layer_id = layer_id + 1
            else:
                print('unknow layer type %s ' % block['type'])
                layer_id = layer_id + 1
        print('save prototxt to %s' % protofile)
        save_prototxt(net_info, protofile, region=False)
        print('save caffemodel to %s' % caffemodel)
        net.save(caffemodel)

    def load_conv2caffe(self, buf, start, conv_param):
        weight = conv_param[0].data
        bias = conv_param[1].data
        conv_param[1].data[...] = np.reshape(buf[start:start + bias.size], bias.shape)
        start = start + bias.size
        conv_param[0].data[...] = np.reshape(buf[start:start + weight.size], weight.shape)
        start = start + weight.size
        return start

    def load_fc2caffe(self, buf, start, fc_param):
        weight = fc_param[0].data
        bias = fc_param[1].data
        fc_param[1].data[...] = np.reshape(buf[start:start + bias.size], bias.shape)
        start = start + bias.size
        fc_param[0].data[...] = np.reshape(buf[start:start + weight.size], weight.shape)
        start = start + weight.size
        return start

    def load_conv_bn2caffe(self, buf, start, conv_param, bn_param, scale_param):
        conv_weight = conv_param[0].data
        running_mean = bn_param[0].data
        running_var = bn_param[1].data
        scale_weight = scale_param[0].data
        scale_bias = scale_param[1].data
        scale_param[1].data[...] = np.reshape(buf[start:start + scale_bias.size], scale_bias.shape)
        start = start + scale_bias.size
        scale_param[0].data[...] = np.reshape(buf[start:start + scale_weight.size], scale_weight.shape)
        start = start + scale_weight.size
        bn_param[0].data[...] = np.reshape(buf[start:start + running_mean.size], running_mean.shape)
        start = start + running_mean.size
        bn_param[1].data[...] = np.reshape(buf[start:start + running_var.size], running_var.shape)
        start = start + running_var.size
        # print(running_var.size)

        bn_param[2].data[...] = np.array([1.0])
        # bn_param[2].data[...] = np.array(buf[start:start + 1]); start = start + 1
        conv_param[0].data[...] = np.reshape(buf[start:start + conv_weight.size], conv_weight.shape)
        start = start + conv_weight.size
        # print(conv_weight.size)

        return start

    def cfg2prototxt(self, cfgfile):
        blocks = parse_cfg(cfgfile)

        prev_filters = 3
        layers = []
        props = OrderedDict()
        bottom = 'data'
        layer_id = 0
        topnames = dict()
        for block in blocks:
            if block['type'] == 'net':
                props['name'] = 'data'
                props['input'] = 'data'
                props['input_dim'] = ['1']
                props['input_dim'].append(block['channels'])
                props['input_dim'].append(block['height'])
                props['input_dim'].append(block['width'])
                continue
            elif block['type'] == 'convolutional':
                conv_layer = OrderedDict()
                conv_layer['bottom'] = bottom
                if 'name' in block:
                    conv_layer['top'] = block['name']
                    conv_layer['name'] = block['name']
                else:
                    conv_layer['top'] = 'layer%d-conv' % layer_id
                    conv_layer['name'] = 'layer%d-conv' % layer_id
                conv_layer['type'] = 'Convolution'
                convolution_param = OrderedDict()
                convolution_param['num_output'] = block['filters']
                prev_filters = block['filters']
                convolution_param['kernel_size'] = block['size']
                if block['pad'] == '1':
                    convolution_param['pad'] = str(int(convolution_param['kernel_size']) // 2)
                convolution_param['stride'] = block['stride']
                if block['batch_normalize'] == '1':
                    convolution_param['bias_term'] = 'false'
                else:
                    convolution_param['bias_term'] = 'true'
                convolution_param['dilation'] = '1'
                # convolution_param['engine'] = 'CAFFE'
                conv_layer['convolution_param'] = convolution_param
                layers.append(conv_layer)
                bottom = conv_layer['top']

                if block['batch_normalize'] == '1':
                    bn_layer = OrderedDict()
                    bn_layer['bottom'] = bottom
                    bn_layer['top'] = bottom
                    if 'name' in block:
                        bn_layer['name'] = '%s-bn' % block['name']
                    else:
                        bn_layer['name'] = 'layer%d-bn' % layer_id
                    bn_layer['type'] = 'BatchNorm'
                    batch_norm_param = OrderedDict()
                    batch_norm_param['use_global_stats'] = 'true'
                    bn_layer['batch_norm_param'] = batch_norm_param
                    layers.append(bn_layer)

                    scale_layer = OrderedDict()
                    scale_layer['bottom'] = bottom
                    scale_layer['top'] = bottom
                    if 'name' in block:
                        scale_layer['name'] = '%s-scale' % block['name']
                    else:
                        scale_layer['name'] = 'layer%d-scale' % layer_id
                    scale_layer['type'] = 'Scale'
                    scale_param = OrderedDict()
                    scale_param['bias_term'] = 'true'
                    scale_layer['scale_param'] = scale_param
                    layers.append(scale_layer)

                if block['activation'] != 'linear':
                    relu_layer = OrderedDict()
                    relu_layer['bottom'] = bottom
                    relu_layer['top'] = bottom
                    if 'name' in block:
                        relu_layer['name'] = '%s-act' % block['name']
                    else:
                        relu_layer['name'] = 'layer%d-act' % layer_id
                    relu_layer['type'] = 'ReLU'
                    if block['activation'] == 'leaky':
                        relu_param = OrderedDict()
                        relu_param['negative_slope'] = '0.1'
                        relu_layer['relu_param'] = relu_param
                    layers.append(relu_layer)
                topnames[layer_id] = bottom
                layer_id = layer_id + 1
            elif block['type'] == 'depthwise_convolutional':
                conv_layer = OrderedDict()
                conv_layer['bottom'] = bottom
                if 'name' in block:
                    conv_layer['top'] = block['name']
                    conv_layer['name'] = block['name']
                else:
                    conv_layer['top'] = 'layer%d-dwconv' % layer_id
                    conv_layer['name'] = 'layer%d-dwconv' % layer_id
                conv_layer['type'] = 'ConvolutionDepthwise'
                convolution_param = OrderedDict()
                convolution_param['num_output'] = prev_filters
                convolution_param['kernel_size'] = block['size']
                if block['pad'] == '1':
                    convolution_param['pad'] = str(int(convolution_param['kernel_size']) // 2)
                convolution_param['stride'] = block['stride']
                if block['batch_normalize'] == '1':
                    convolution_param['bias_term'] = 'false'
                else:
                    convolution_param['bias_term'] = 'true'
                convolution_param['dilation'] = '1'
                # convolution_param['engine'] = 'CAFFE'
                conv_layer['convolution_param'] = convolution_param
                layers.append(conv_layer)
                bottom = conv_layer['top']

                if block['batch_normalize'] == '1':
                    bn_layer = OrderedDict()
                    bn_layer['bottom'] = bottom
                    bn_layer['top'] = bottom
                    if 'name' in block:
                        bn_layer['name'] = '%s-bn' % block['name']
                    else:
                        bn_layer['name'] = 'layer%d-bn' % layer_id
                    bn_layer['type'] = 'BatchNorm'
                    batch_norm_param = OrderedDict()
                    batch_norm_param['use_global_stats'] = 'true'
                    bn_layer['batch_norm_param'] = batch_norm_param
                    layers.append(bn_layer)

                    scale_layer = OrderedDict()
                    scale_layer['bottom'] = bottom
                    scale_layer['top'] = bottom
                    if 'name' in block:
                        scale_layer['name'] = '%s-scale' % block['name']
                    else:
                        scale_layer['name'] = 'layer%d-scale' % layer_id
                    scale_layer['type'] = 'Scale'
                    scale_param = OrderedDict()
                    scale_param['bias_term'] = 'true'
                    scale_layer['scale_param'] = scale_param
                    layers.append(scale_layer)

                if block['activation'] != 'linear':
                    relu_layer = OrderedDict()
                    relu_layer['bottom'] = bottom
                    relu_layer['top'] = bottom
                    if 'name' in block:
                        relu_layer['name'] = '%s-act' % block['name']
                    else:
                        relu_layer['name'] = 'layer%d-act' % layer_id
                    relu_layer['type'] = 'ReLU'
                    if block['activation'] == 'leaky':
                        relu_param = OrderedDict()
                        relu_param['negative_slope'] = '0.1'
                        relu_layer['relu_param'] = relu_param
                    layers.append(relu_layer)
                topnames[layer_id] = bottom
                layer_id = layer_id + 1
            elif block['type'] == 'maxpool':
                max_layer = OrderedDict()
                max_layer['bottom'] = bottom
                if 'name' in block:
                    max_layer['top'] = block['name']
                    max_layer['name'] = block['name']
                else:
                    max_layer['top'] = 'layer%d-maxpool' % layer_id
                    max_layer['name'] = 'layer%d-maxpool' % layer_id
                max_layer['type'] = 'Pooling'
                pooling_param = OrderedDict()
                pooling_param['stride'] = block['stride']
                pooling_param['pool'] = 'MAX'
                if (int(block['size']) - int(block['stride'])) % 2 == 0:
                    pooling_param['kernel_size'] = block['size']
                    pooling_param['pad'] = str((int(block['size']) - 1) // 2)

                if (int(block['size']) - int(block['stride'])) % 2 == 1:
                    pooling_param['kernel_size'] = str(int(block['size']) + 1)
                    pooling_param['pad'] = str((int(block['size']) + 1) // 2)

                max_layer['pooling_param'] = pooling_param
                layers.append(max_layer)
                bottom = max_layer['top']
                topnames[layer_id] = bottom
                layer_id = layer_id + 1
            elif block['type'] == 'avgpool':
                avg_layer = OrderedDict()
                avg_layer['bottom'] = bottom
                if 'name' in block:
                    avg_layer['top'] = block['name']
                    avg_layer['name'] = block['name']
                else:
                    avg_layer['top'] = 'layer%d-avgpool' % layer_id
                    avg_layer['name'] = 'layer%d-avgpool' % layer_id
                avg_layer['type'] = 'Pooling'
                pooling_param = OrderedDict()
                pooling_param['kernel_size'] = 7
                pooling_param['stride'] = 1
                pooling_param['pool'] = 'AVE'
                avg_layer['pooling_param'] = pooling_param
                layers.append(avg_layer)
                bottom = avg_layer['top']
                topnames[layer_id] = bottom
                layer_id = layer_id + 1
            elif block['type'] == 'route':
                from_layers = block['layers'].split(',')
                bottom_tmp = []
                if len(from_layers) == 1:
                    if int(from_layers[0]) < 0:
                        prev_layer_id = layer_id + int(from_layers[0])
                    else:
                        prev_layer_id = int(from_layers[0])
                    bottom = topnames[prev_layer_id]
                    topnames[layer_id] = bottom
                    layer_id = layer_id + 1
                else:
                    for index in range(len(from_layers)):
                        if int(from_layers[index]) < 0:
                            prev_layer_id = layer_id + int(from_layers[index])
                        else:
                            prev_layer_id = int(from_layers[index])
                        bottom = topnames[prev_layer_id]
                        bottom_tmp.append(bottom)
                    concat_layer = OrderedDict()
                    concat_layer['bottom'] = bottom_tmp
                    if 'name' in block:
                        concat_layer['top'] = block['name']
                        concat_layer['name'] = block['name']
                    else:
                        concat_layer['top'] = 'layer%d-concat' % layer_id
                        concat_layer['name'] = 'layer%d-concat' % layer_id
                    concat_layer['type'] = 'Concat'
                    layers.append(concat_layer)
                    bottom = concat_layer['top']
                    topnames[layer_id] = bottom
                    layer_id = layer_id + 1

            elif block['type'] == 'shortcut':
                prev_layer_id1 = layer_id + int(block['from'])
                prev_layer_id2 = layer_id - 1
                bottom1 = topnames[prev_layer_id1]
                bottom2 = topnames[prev_layer_id2]
                shortcut_layer = OrderedDict()
                shortcut_layer['bottom'] = [bottom1, bottom2]
                if 'name' in block:
                    shortcut_layer['top'] = block['name']
                    shortcut_layer['name'] = block['name']
                else:
                    shortcut_layer['top'] = 'layer%d-shortcut' % layer_id
                    shortcut_layer['name'] = 'layer%d-shortcut' % layer_id
                shortcut_layer['type'] = 'Eltwise'
                eltwise_param = OrderedDict()
                eltwise_param['operation'] = 'SUM'
                shortcut_layer['eltwise_param'] = eltwise_param
                layers.append(shortcut_layer)
                bottom = shortcut_layer['top']

                if block['activation'] != 'linear':
                    relu_layer = OrderedDict()
                    relu_layer['bottom'] = bottom
                    relu_layer['top'] = bottom
                    if 'name' in block:
                        relu_layer['name'] = '%s-act' % block['name']
                    else:
                        relu_layer['name'] = 'layer%d-act' % layer_id
                    relu_layer['type'] = 'ReLU'
                    if block['activation'] == 'leaky':
                        relu_param = OrderedDict()
                        relu_param['negative_slope'] = '0.1'
                        relu_layer['relu_param'] = relu_param
                    layers.append(relu_layer)
                topnames[layer_id] = bottom
                layer_id = layer_id + 1

            elif block['type'] == 'connected':
                fc_layer = OrderedDict()
                fc_layer['bottom'] = bottom
                if 'name' in block:
                    fc_layer['top'] = block['name']
                    fc_layer['name'] = block['name']
                else:
                    fc_layer['top'] = 'layer%d-fc' % layer_id
                    fc_layer['name'] = 'layer%d-fc' % layer_id
                fc_layer['type'] = 'InnerProduct'
                fc_param = OrderedDict()
                fc_param['num_output'] = int(block['output'])
                fc_layer['inner_product_param'] = fc_param
                layers.append(fc_layer)
                bottom = fc_layer['top']

                if block['activation'] != 'linear':
                    relu_layer = OrderedDict()
                    relu_layer['bottom'] = bottom
                    relu_layer['top'] = bottom
                    if 'name' in block:
                        relu_layer['name'] = '%s-act' % block['name']
                    else:
                        relu_layer['name'] = 'layer%d-act' % layer_id
                    relu_layer['type'] = 'ReLU'
                    if block['activation'] == 'leaky':
                        relu_param = OrderedDict()
                        relu_param['negative_slope'] = '0.1'
                        relu_layer['relu_param'] = relu_param
                    layers.append(relu_layer)
                topnames[layer_id] = bottom
                layer_id = layer_id+1

            elif block['type'] == 'upsample':
                deconv_layer = OrderedDict()
                stride = block['stride']
                print("stride: %s" % stride)
                if 'name' in block:
                    deconv_layer['name'] = block['name']
                    deconv_layer['type'] = 'Deconvolution'
                    deconv_layer['bottom'] = bottom
                    deconv_layer['top'] = block['name']
                else:
                    deconv_layer['name'] = 'layer%d-deconv' % layer_id
                    deconv_layer['type'] = 'Deconvolution'
                    deconv_layer['bottom'] = bottom
                    deconv_layer['top'] = 'layer%d-deconv' % layer_id
                print("num_output: %s" % conv_layer['convolution_param']['num_output'])
                convolution_param = OrderedDict()
                convolution_param['num_output'] = conv_layer['convolution_param']['num_output']
                convolution_param['group'] = conv_layer['convolution_param']['num_output']
                convolution_param['kernel_size'] = 2 * int(stride) - int(stride) % 2
                convolution_param['stride'] = stride
                convolution_param['pad'] = int(np.ceil((int(stride) - 1) / 2.0))
                convolution_param['bias_term'] = 'false'
                weight_filler_param = OrderedDict()
                weight_filler_param['type'] = 'bilinear'
                deconv_layer['convolution_param'] = convolution_param
                convolution_param['weight_filler'] = weight_filler_param
                layers.append(deconv_layer)
                topnames[layer_id] = deconv_layer['top']
                layer_id = layer_id + 1

            # add reorg layer yolov2
            elif block['type'] == 'reorg':
                reorg_layer = OrderedDict()
                reorg_layer['bottom'] = bottom
                if 'name' in block:
                    reorg_layer['top'] = block['name']
                    reorg_layer['name'] = block['name']
                else:
                    reorg_layer['top'] = 'layer%d-reorg' % layer_id
                    reorg_layer['name'] = 'layer%d-reorg' % layer_id
                reorg_layer['type'] = 'Reorg'
                reorg_param = OrderedDict()
                reorg_param['stride'] = 2
                reorg_layer['reorg_param'] = reorg_param
                layers.append(reorg_layer)
                bottom = reorg_layer['top']
                topnames[layer_id] = bottom
                layer_id = layer_id + 1
            elif block['type'] == 'region':
                if True:
                    region_layer = OrderedDict()
                    region_layer['bottom'] = bottom
                    if 'name' in block:
                        region_layer['top'] = block['name']
                        region_layer['name'] = block['name']
                    else:
                        region_layer['top'] = 'layer%d-region' % layer_id
                        region_layer['name'] = 'layer%d-region' % layer_id
                    region_layer['type'] = 'Region'
                    region_param = OrderedDict()
                    region_param['anchors'] = block['anchors'].strip()
                    region_param['classes'] = block['classes']
                    region_param['num'] = block['num']
                    region_layer['region_param'] = region_param
                    layers.append(region_layer)
                    bottom = region_layer['top']
                topnames[layer_id] = bottom
                layer_id = layer_id + 1

            elif block['type'] == 'yolo':
                layer_id = layer_id + 1
            else:
                print('unknow layer type %s ' % block['type'])
                topnames[layer_id] = bottom
                layer_id = layer_id + 1

        net_info = OrderedDict()
        net_info['props'] = props
        net_info['layers'] = layers
        return net_info


if __name__ == '__main__':
    print("process start...")
    options = parse_arguments()
    # net_info = cfg2prototxt(options.cfg_path)
    # print_prototxt(net_info)
    # save_prototxt(net_info, 'tmp.prototxt')
    convert = DarknetConvertCaffe(options.cfg_path, options.weight_path)
    convert.convert_caffe()
    print("process end!")
