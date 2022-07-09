# resnet18 show
python3 easyai/inference_task.py -t classify -i /home/wfw/data/VOCdevkit/caltech100_classify/ImageSets/val.txt \
                                 -m ./cfg/cls/resnet_classify.cfg -w ./log/snapshot/cls_best.pt -c ./log/config/classify_config.json -s

# resnet18 without show
# python3 easyai/inference_task.py -t classify -i /home/wfw/data/VOCdevkit/caltech100_classify/ImageSets/val.txt \
                                 -m ./cfg/cls/resnet_classify.cfg -w ./log/snapshot/cls_best.pt -c ./log/config/classify_config.json
