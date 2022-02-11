import os
import sys
import torch
import codecs, json
import numpy as np
sys.path.insert(0, os.getcwd() + "/..")

from easyai.model.utility.model_factory import ModelFactory


def main(cfgPath, modelPath, savePath):
    model_factory = ModelFactory()
    model_config = {"type": "FairMOTNet",
                    "data_channel": 3,
                    "class_number": 1,
                    "reid": 64}
    model = model_factory.get_model(model_config)
    model_dict = model.state_dict()
    # obj_text = codecs.open('/home/lpj/Downloads/dict.json', 'r', encoding='utf-8').read()
    # b_new = json.loads(obj_text)

    modelCvt = torch.load('/home/lpj/github/FairMOT/models/fairmot_lite.pth',
                          map_location='cpu')
    if isinstance(modelCvt, dict):
        modelCvt = modelCvt['state_dict']
    # modelCvt = np.load('/home/lpj/Downloads/obilenetv3_dict.npy', allow_pickle=True)
    vs = []

    for i, (k, v) in enumerate(modelCvt.items()):
        if "anchor" in k:
            continue
        print(k, v.shape)
        vs.append(v)
    print(len(vs))

    vs_ = []
    index = 0
    for j, (k_, v_) in enumerate(model.state_dict().items()):
        # if "num_batches_tracked" in k_:
        #     continue
        print(k_, v_.shape)
        if j == len(vs):
            break
        v_.copy_(vs[index])
        # vs_.append(v_)
        index += 1
    print(len(vs_))

    # model_dict.update(modelCvt.items())
    # model.load_state_dict(model_dict)
    checkpoint = {'epoch': 0,
                  'best_value': 0,
                  'model': model.state_dict()}
    torch.save(checkpoint, "/home/lpj/fairmot_person.pt")

    print("End of game!!!")


if __name__ == '__main__':
    #options = parse_arguments()
    #main(options.cfg)
    main("./cfg/yolov3-spp-dilation_BerkeleyAll.cfg", "./snapshot/detect/pretain.pt", "./vgg16_FgSegNetV2_2.pt")