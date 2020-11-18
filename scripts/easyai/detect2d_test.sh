# yolov3
python3 easyai/test_task.py -t detect2d -v /home/wfw/data/VOCdevkit/Fruit_detection/ImageSets/val.txt \
                             -m detnet -w ./log/snapshot/det2d_best.pt -c ./log/config/detection2d_config.json
# yolov2
# python3 easyai/test_task.py -t detect2d -v /home/wfw/data/VOCdevkit/Fruit_detection/ImageSets/val.txt \
                             -m ./cfg/det2d/yolov2.cfg -w ./log/snapshot/det2d_best.pt -c ./log/config/detection2d_config.json