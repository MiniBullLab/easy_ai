import os
import cv2
import json
import codecs
import numpy as np
from optparse import OptionParser


def parse_arguments():
    parser = OptionParser()
    parser.description = "This program transform xml to json"

    parser.add_option("-i", "--input_path", dest="input_path",
                      type="string", default=None,
                      help="input path of dataset")

    (options, args) = parser.parse_args()

    if options.input_path:
        if not os.path.exists(options.input_path):
            parser.error("Could not find the input path of dataset")
        else:
            options.input_path = os.path.normpath(options.input_path)
    else:
        parser.error("'input_path' option is required to run this program")

    return options


def parse_rect_data(json_path):
    if not os.path.exists(json_path):
        print("error:%s file not exists" % json_path)
        return
    with codecs.open(json_path, 'r', encoding='utf-8') as f:
        data_dict = json.load(f)
    image_name = data_dict['filename']
    objects_dict = data_dict['objects']
    rect_objects_list = objects_dict['rectObject']
    boxes = []
    for rect_dict in rect_objects_list:
        class_name = rect_dict['class']
        xmin = rect_dict['minX']
        ymin = rect_dict['minY']
        xmax = rect_dict['maxX']
        ymax = rect_dict['maxY']
        box = []
        box.append(class_name)
        box.append(xmin)
        box.append(ymin)
        box.append(xmax)
        box.append(ymax)
        if (xmax-xmin) >= 0 \
                and (ymax-ymin) >= 0:
            boxes.append(box)
    return image_name, boxes


def box_count(input_path):
    class_json_file = os.path.join(input_path, "class.json")
    class_file = open(class_json_file, 'r')
    classes_ = json.load(class_file)
    classes = [value for key, value in classes_.items()]
    num_classes = len(classes)
    box_cal = np.zeros(num_classes, dtype='int')
    for img_name in os.listdir(os.path.join(input_path, "JPEGImages")):
        img_path = os.path.join(input_path, "JPEGImages", img_name)
        json_path = img_path.replace("JPEGImages", "Annotations").replace(".jpg", ".json").replace(".png", ".json")
        if not os.path.exists(json_path):
            continue
        image_name, boxes = parse_rect_data(json_path)
        for box in boxes:
            cls = classes.index(box[0])
            box_cal[cls] += 1

    return classes, box_cal


def main():
    print("process start...")
    options = parse_arguments()
    classes, box_cal = box_count(options.input_path)

    print("The distribution is: \n")
    for c in classes:
        cls_index = classes.index(c)
        print("{} box num is {}".format(c, box_cal[cls_index]))
    print("End of game, have a nice day!")


if __name__ == "__main__":
    main()