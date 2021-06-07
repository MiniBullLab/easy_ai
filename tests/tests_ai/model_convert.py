import os
import sys
import torch
import codecs, json
import numpy as np
sys.path.insert(0, os.getcwd() + "/..")

from easyai.model.utility.model_factory import ModelFactory

def main(cfgPath, modelPath, savePath):
    model_factory = ModelFactory()
    model_config = {"type": 'dbnet'}
    model = model_factory.get_model(model_config)
    obj_text = codecs.open('/home/lpj/Downloads/dict.json', 'r', encoding='utf-8').read()
    b_new = json.loads(obj_text)

    # modelCvt = torch.load('/home/lpj/Downloads/mobilenetv3_crnn.pth', map_location='cpu')
    # print(modelCvt.dtype)
    # if isinstance(modelCvt, dict):
    #     modelCvt = modelCvt['model']
    # modelCvt = np.load('/home/lpj/Downloads/obilenetv3_dict.npy', allow_pickle=True)
    vs = []

    for i, (k, v) in enumerate(b_new.items()):
        print(k)
        vs.append(np.array(v))
    print(len(vs))

    vs_ = []
    for j, (k_, v_) in enumerate(model.state_dict().items()):
        vs_.append(v_)
    print(len(vs_))
        # if j < 21:
        #     print(j, k_, vs[j].shape, v_.shape)
        #     v_.copy_(vs[j])

    # checkpoint = {'epoch': None,
    #               'best_value': None,
    #               'model': model.state_dict(),
    #               'optimizer': None}
    # torch.save(checkpoint, savePath)

    print("End of game!!!")


if __name__ == '__main__':
    #options = parse_arguments()
    #main(options.cfg)
    main("./cfg/yolov3-spp-dilation_BerkeleyAll.cfg", "./snapshot/detect/pretain.pt", "/home/wfw/HASCO/all_wights/vgg16_FgSegNetV2_2.pt")