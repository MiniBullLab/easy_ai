# yolov3 show
python3 easyai/inference_task.py -t detect2d -i /home/wfw/data/VOCdevkit/Fruit_detection/ImageSets/val.txt \
                                 -m detnet -w ./log/snapshot/det2d_best.pt -c ./log/config/detection2d_config.json -s

# yolov3 without show
# python3 easyai/inference_task.py -t detect2d -i /home/wfw/data/VOCdevkit/Fruit_detection/ImageSets/val.txt \
                                 -m detnet -w ./log/snapshot/det2d_best.pt -c ./log/config/detection2d_config.json
