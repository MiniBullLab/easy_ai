import os
import glob
import json
import codecs
import cv2
import numpy as np

provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']


def STRcoordinates_deal(STRcoordinates, i):
    result = STRcoordinates[i]
    x, y = int(result.split('&')[0]), int(result.split('&')[1])
    result = x, y
    return result


def analyse_file(filename):
    STRinformation = filename.split('-')
    STRarea_ratio = STRinformation[0]
    STRangle = STRinformation[1]
    # 386&473_177&454_154&383_363&402 右下、左下、左上、右下
    STRcoordinates = STRinformation[3]
    STRchars = STRinformation[4]

    # 坐标处理
    coordinates = []
    STRcoordinates = STRcoordinates.split('_')
    right_bottom = STRcoordinates_deal(STRcoordinates, 0)
    coordinates.append(right_bottom[0])
    coordinates.append(right_bottom[1])
    left_bottom = STRcoordinates_deal(STRcoordinates, 1)
    coordinates.append(left_bottom[0])
    coordinates.append(left_bottom[1])
    left_top = STRcoordinates_deal(STRcoordinates, 2)
    coordinates.append(left_top[0])
    coordinates.append(left_top[1])
    right_top = STRcoordinates_deal(STRcoordinates, 3)
    coordinates.append(right_top[0])
    coordinates.append(right_top[1])

    # 车牌字符处理
    LPchars = []
    STRchars = STRchars.split('_')
    LPchars.append(provinces[int(STRchars[0])])
    LPchars.append(alphabets[int(STRchars[1])])
    for i in range(2, 7):
        LPchars.append(ads[int(STRchars[i])])
    resultText = ''.join(LPchars)
    return coordinates, resultText


def save_ocr_data(file_name, file_path, image_size, data_list):
    path, file_name_and_post = os.path.split(file_path)
    image_name, post = os.path.splitext(file_name_and_post)
    json_dir = os.path.join(path, "../Annotations")
    if not os.path.exists(json_dir):
        os.mkdir(json_dir)
    json_path = os.path.join(json_dir, image_name + ".json")
    annotation = dict()
    # annotation
    annotation['annotation'] = 'Annotations'
    # database
    annotation['database'] = "CCPD"
    # owner
    annotation['owner'] = 'miniBull'
    # folder
    annotation['folder'] = 'JPEGImages'
    # filename
    annotation['filename'] = file_name
    # path
    annotation['path'] = file_path
    # size
    annotation['size'] = {'width': image_size[0],
                          'height': image_size[1],
                          'depth': 3}
    # objectCount
    annotation['objectCount'] = len(data_list)
    # objects
    ocrObject = []
    for coordinates, resultText in data_list:
        temp_dict = dict()
        temp_dict['class'] = 'others'
        temp_dict['illegibility'] = 0
        temp_dict['language'] = 'chinese'
        temp_dict['pointCount'] = len(coordinates) // 2
        temp_dict['polygon'] = coordinates
        temp_dict['transcription'] = resultText
        ocrObject.append(temp_dict)
    annotation['objects'] = {'ocrObject': ocrObject}

    with codecs.open(json_path, 'w', encoding='utf-8') as f:
        json.dump(annotation, f, sort_keys=True, indent=4, ensure_ascii=False)


def getDirFiles(dataDir, filePost):
    imagePathPattern = os.path.join(dataDir, filePost)
    for filePath in glob.iglob(imagePathPattern):
        yield filePath
    return


def CCPD_process(data_dir):
    index = 0
    for imagePath in getDirFiles(data_dir, "*.jpg"):
        print(imagePath)
        src_image = cv2.imdecode(np.fromfile(imagePath, dtype=np.uint8),
                                 cv2.IMREAD_COLOR)
        if src_image is not None:
            shape = src_image.shape[:2]  # shape = [height, width]
            image_size = (shape[1], shape[0])
            path, file_name_and_post = os.path.split(imagePath)
            image_name, post = os.path.splitext(file_name_and_post)
            information = file_name_and_post.split('-')
            print(information)
            file_name = information[0]
            file_name = "%d_%s.jpg" % (index, file_name)
            image_dir = os.path.join(path, "../JPEGImages")
            if not os.path.exists(image_dir):
                os.mkdir(image_dir)
            file_path = os.path.join(image_dir, file_name)
            os.rename(imagePath, file_path)
            print(file_path)
            result = analyse_file(image_name)
            save_ocr_data(file_name, file_path, image_size, (result, ))
            index += 1


if __name__ == "__main__":
    CCPD_process("/home/lpj/dataset/CCPD2019/ccpd_weather")