# mobilenetv2_fgseg
python3 easyai/train_task.py -t segment -i /home/wfw/data/VOCdevkit/CarScratch_segment/ImageSets/train.txt -v /home/wfw/data/VOCdevkit/CarScratch_segment/ImageSets/val.txt \
                             -m ./cfg/seg/mobilenetv2_fgseg.cfg -p /home/wfw/HASCO/all_wights/MobilenetV2_FgSegNetV2.pt -c ./log/config/segmention_config.json
# fgsegv2
# python3 easyai/train_task.py -t segment -i /home/wfw/data/VOCdevkit/CarScratch_segment/ImageSets/train.txt -v /home/wfw/data/VOCdevkit/CarScratch_segment/ImageSets/val.txt \
                             -m ./cfg/seg/fgsegv2_new.cfg.cfg -p /home/wfw/HASCO/all_wights/vgg16_FgSegNetV2.pt -c ./log/config/segmention_config.json
# mobilenetv2_fcn
# python3 easyai/train_task.py -t segment -i /home/wfw/data/VOCdevkit/CarScratch_segment/ImageSets/train.txt -v /home/wfw/data/VOCdevkit/CarScratch_segment/ImageSets/val.txt \
                             -m ./cfg/seg/mobilenet_fcn.cfg -p /home/wfw/HASCO/all_wights/mobilenetv2_FCN.pt -c ./log/config/segmention_config.json