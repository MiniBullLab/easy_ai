# mobilenetv2_fgseg
python3 easyai/test_task.py -t segment -v /home/wfw/data/VOCdevkit/CarScratch_segment/ImageSets/val.txt \
                             -m ./cfg/seg/mobilenetv2_fgseg.cfg -w ./log/snapshot/seg_best.pt -c ./log/config/segmention_config.json
# fgsegv2
# python3 easyai/test_task.py -t segment -v /home/wfw/data/VOCdevkit/CarScratch_segment/ImageSets/val.txt \
                             -m ./cfg/seg/fgsegv2_new.cfg.cfg -w ./log/snapshot/seg_best.pt -c ./log/config/segmention_config.json
# mobilenetv2_fcn
# python3 easyai/test_task.py -t segment -v /home/wfw/data/VOCdevkit/CarScratch_segment/ImageSets/val.txt \
                             -m ./cfg/seg/mobilenet_fcn.cfg -w ./log/snapshot/seg_best.pt -c ./log/config/segmention_config.json