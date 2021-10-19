# yolov3
python3 easyai/train_task.py -t detect2d -i /home/wfw/data/VOCdevkit/Fruit_detection/ImageSets/train.txt -v /home/wfw/data/VOCdevkit/Fruit_detection/ImageSets/val.txt \
                             -m detnet -p /home/wfw/HASCO/all_wights/yolov3_coco.pt -c ./log/config/detection2d_config.json
# yolov2
# python3 easyai/train_task.py -t detect2d -i /home/wfw/data/VOCdevkit/Fruit_detection/ImageSets/train.txt -v /home/wfw/data/VOCdevkit/Fruit_detection/ImageSets/val.txt \
                             -m ./cfg/det2d/yolov2.cfg -p /home/wfw/HASCO/all_wights/yolov2_coco.pt -c ./log/config/detection2d_config.json
# shufflenetv2-yolov3
# python3 easyai/train_task.py -t detect2d -i /home/wfw/data/VOCdevkit/Fruit_detection/ImageSets/train.txt -v /home/wfw/data/VOCdevkit/Fruit_detection/ImageSets/val.txt \
                             -m ./cfg/det2d/shufflenetV2-0.5_spp_BerkeleyAll.cfg -p /home/wfw/HASCO/all_wights/shufflenetV2_yolov3.pt -c ./log/config/detection2d_config.json
# ssd512
# python3 easyai/train_task.py -t detect2d -i /home/wfw/data/VOCdevkit/Fruit_detection/ImageSets/train.txt -v /home/wfw/data/VOCdevkit/Fruit_detection/ImageSets/val.txt \
                             -m ./cfg/det2d/ssd512.cfg -p /home/wfw/HASCO/all_wights/vgg16_ssd512.pt -c ./log/config/detection2d_config.json