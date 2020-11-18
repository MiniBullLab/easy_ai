# resnet18
python3 easyai/test_task.py -t classify -v /home/wfw/data/VOCdevkit/caltech100_classify/ImageSets/val.txt \
                             -m ./cfg/cls/resnet_classify.cfg -w ./log/snapshot/cls_best.pt -c ./log/config/classify_config.json

# mobilenetv2
# python3 easyai/train_task.py -t classify -v /home/wfw/data/VOCdevkit/caltech100_classify/ImageSets/val.txt \
                             -m ./cfg/cls/mobilenetv2_cls.cfg -p ./log/snapshot/cls_best.pt -c ./log/config/classify_config.json
