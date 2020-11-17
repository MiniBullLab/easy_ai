import torch
from easyai.base_name.model_name import ModelName
from easyai.model.utility.model_factory import ModelFactory
from easyai.torch_utility.torch_onnx.model_show import ModelShow


def main():
    model_factory = ModelFactory()
    show_process = ModelShow()
    image = torch.randn(1, 1, 28, 28)
    model_config = {"type": ModelName.MNISTGan,
                    'data_channel': 1,
                    'image_size': (28, 28)}
    model = model_factory.get_model(model_config)
    input_x = model.generator_input_data(image, 1)
    show_process.set_input(input_x)
    show_process.show_from_model(model)


if __name__ == '__main__':
    main()

