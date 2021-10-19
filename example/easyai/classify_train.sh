# resnet18
python3 easyai/train_task.py -t classify -i /home/wfw/data/VOCdevkit/caltech100_classify/ImageSets/train.txt -v /home/wfw/data/VOCdevkit/caltech100_classify/ImageSets/val.txt \
                             -m ./cfg/cls/resnet_classify.cfg -p /home/wfw/HASCO/all_wights/resnet18_224.pt -c ./log/config/classify_config.json

# mobilenetv2
# python3 easyai/train_task.py -t classify -i /home/wfw/data/VOCdevkit/caltech100_classify/ImageSets/train.txt -v /home/wfw/data/VOCdevkit/caltech100_classify/ImageSets/val.txt \
                             -m ./cfg/cls/mobilenetv2_cls.cfg -p /home/wfw/HASCO/all_wights/mobilenetv2_224.pt -c ./log/config/classify_config.json
