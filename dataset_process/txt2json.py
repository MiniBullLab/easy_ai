import os
import cv2
import json
import numpy as np
import xml.dom
import xml.dom.minidom
import xml.etree.ElementTree as ElementTree
from optparse import OptionParser


def parse_arguments():
    parser = OptionParser()
    parser.description = "This program transform xml to json"

    parser.add_option("-i", "--inputPath", dest="inputPath",
                      type="string", default=None,
                      help="input path of dataset")

    (options, args) = parser.parse_args()

    if options.inputPath:
        if not os.path.exists(options.inputPath):
            parser.error("Could not find the input path of dataset")
        else:
            options.input_path = os.path.normpath(options.inputPath)
    else:
        parser.error("'inputPath' option is required to run this program")

    return options

def Deconvert(size, box):
    w = size[0]
    h = size[1]
    x = float(box[0])*w
    y = float(box[1])*h
    w = float(box[2])*w
    h = float(box[3])*h
    bx1 = int(((x+1)*2.0-w)/2.0)
    bx2 = int(((x+1)*2.0+w)/2.0)
    by1 = int(((y+1)*2.0-h)/2.0)
    by2 = int(((y+1)*2.0+h)/2.0)

    return bx1,bx2,by1,by2

def parseRectData(xmlPath):
    xmlTree = ElementTree.parse(xmlPath)
    root = xmlTree.getroot()
    folder_node = root.find("folder")
    folder = folder_node.text
    image_node = root.find("filename")
    image_name = image_node.text
    # print(image_name)
    imageSizeNode = root.find("size")
    widthNode = imageSizeNode.find("width")
    heightNode = imageSizeNode.find("height")
    depthNode = imageSizeNode.find("depth")
    image_size = [int(widthNode.text), int(heightNode.text), int(depthNode.text)]

    num = 0
    boxes = []
    for object_node in root.findall('rectObject'):
        num += 1
        box_node = object_node.find("bndbox")
        name_node = object_node.find("name")
        xMinNode = box_node.find("xmin")
        yMinNode = box_node.find("ymin")
        xMaxNode = box_node.find("xmax")
        yMaxNode = box_node.find("ymax")
        xMin = int(xMinNode.text)
        yMin = int(yMinNode.text)
        xMax = int(xMaxNode.text)
        yMax = int(yMaxNode.text)
        box = []
        box.append(name_node.text)
        box.append(xMin)
        box.append(yMin)
        box.append(xMax)
        box.append(yMax)
        boxes.append(box)
    return image_name, image_size, boxes, num

def parseTxtData(txtPath, image_size):
    file_p = open(txtPath, "r")
    file_rects = file_p.readlines()

    num = 0
    boxes = []
    for file_rect in file_rects:
        num += 1
        rect_ = file_rect[:-1].split(" ")
        xMin, xMax, yMin, yMax = Deconvert(image_size, (float(rect_[2]), float(rect_[3]), float(rect_[4]), float(rect_[5])))

        box = []
        box.append(int(rect_[1]))
        box.append("person")
        box.append(xMin)
        box.append(yMin)
        box.append(xMax)
        box.append(yMax)
        boxes.append(box)
        
    return image_size, boxes, num

def json_write(database, file_name, file_path, image_size, num, boxes):
    annotation = {}
    # annotation
    annotation['annotation'] = 'Annotations'
    # database
    annotation['database'] = database
    # owner
    annotation['owner'] = 'miniBull'
    # folder
    annotation['folder'] = 'JPEGImages'
    # filename
    annotation['filename'] = file_name
    # path
    annotation['path'] = file_path
    # size
    annotation['size'] = {'width': image_size[0], 'height': image_size[1], 'depth': image_size[2]}
    # objectCount
    annotation['objectCount'] = num
    # objects
    rectObject = []
    for box in boxes:
        rectObject.append({'id': box[0], 'class': box[1], 'minX': box[2], 'minY': box[3], 'maxX': box[4], 'maxY': box[5]})
    annotation['objects'] = {'rectObject': rectObject}

    json_path = file_path.replace("images", "Annotations_json").replace("jpg", "json").replace("png", "json")
    a = json.dumps(annotation, indent=4)
    f = open(json_path, 'w')
    f.write(a)
    f.close()


def main():
    print("process start...")
    options = parse_arguments()
    # if not os.path.exists(options.inputPath.replace("images", "Annotations_json")):
    #     os.mkdir(options.inputPath.replace("images", "Annotations_json"))
    for dir_name in os.listdir(options.inputPath):
        img_path = os.path.join(options.inputPath, dir_name)
        print(os.path.join(options.inputPath, dir_name).replace("images", "Annotations_json"))
        if not os.path.exists(os.path.join(options.inputPath, dir_name).replace("images", "Annotations_json")):
            os.mkdir(os.path.join(options.inputPath, dir_name).replace("images", "Annotations_json"))
            os.mkdir(os.path.join(options.inputPath, dir_name, "img1").replace("images", "Annotations_json"))
        for img_name in os.listdir(os.path.join(options.inputPath, dir_name, "img1")):
            img_path = os.path.join(options.inputPath, dir_name, "img1", img_name)
            img = cv2.imread(img_path)
            image_size = (img.shape[1], img.shape[0], img.shape[2])
            xml_path = img_path.replace("images", "labels_with_ids").replace(".jpg", ".txt").replace(".png", ".txt")
            if not os.path.exists(xml_path):
                continue
             
            image_size, boxes, num = parseTxtData(xml_path, image_size)
            json_write("DataSet", img_name, img_path, image_size, num, boxes)


if __name__ == "__main__":
    main()
