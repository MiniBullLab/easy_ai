import os
from optparse import OptionParser
import pathlib
import onnx
import onnxmltools
from keras2onnx import convert_keras
from keras import models


def parse_arguments():
    parser = OptionParser()
    parser.description = "This program is keras convert to onnx"

    parser.add_option("-i", "--h5", dest="h5_path",
                      metavar="PATH", type="string", default=None,
                      help="keras h5 model path")

    parser.add_option("-m", "--model", dest="model_name",
                      type="string", default=None,
                      help="keras model name")

    (options, args) = parser.parse_args()

    return options


class KerasConvertOnnx():

    def __init__(self, h5_model_path):
        self.target_opset = 10
        self.h5_model_path = pathlib.Path(h5_model_path)
        self.onnx_save_path = self.h5_model_path.with_suffix(".onnx")

    def convert_onnx_from_h5(self, net_name):
        # get model struct and weights
        keras_model = models.load_model(str(self.h5_model_path))
        # onnx_model = onnxmltools.convert_keras(keras_model)
        onnx_model = convert_keras(keras_model, net_name,
                                   target_opset=self.target_opset,
                                   channel_first_inputs=['net_input'])
        onnx.save_model(onnx_model, str(self.onnx_save_path))


def main():
    print("process start...")
    options = parse_arguments()
    converter = KerasConvertOnnx(options.h5_path)
    converter.convert_onnx_from_h5(options.model_name)
    print("process end!")


if __name__ == "__main__":
    main()

