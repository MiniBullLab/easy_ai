# mobilenetv2_fgseg show
python3 easyai/inference_task.py -t segment -v /home/wfw/data/VOCdevkit/CarScratch_segment/ImageSets/val.txt \
                                 -m ./cfg/seg/mobilenetv2_fgseg.cfg -w ./log/snapshot/seg_best.pt -c ./log/config/segmention_config.json -s

# mobilenetv2_fgseg without show
# python3 easyai/inference_task.py -t segment -v /home/wfw/data/VOCdevkit/CarScratch_segment/ImageSets/val.txt \
                                 -m ./cfg/seg/mobilenetv2_fgseg.cfg -w ./log/snapshot/seg_best.pt -c ./log/config/segmention_config.json
