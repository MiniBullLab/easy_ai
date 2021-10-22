import pickle
import os
from os import listdir, getcwd
from os.path import join
import json
import argparse


classes = []
#该函数用于遍历所有的标注返回全部的待检测类别
def gen_classes(json_path):
    files= os.listdir(json_path)
    for in_file in files:
        #判断是不是文件夹
        temp_path = os.path.join(json_path, in_file)
        if not os.path.isdir(temp_path):
            #print("file is :", json_path +"/"+ in_file)
            #下面两行是验证过的json打开方式，注意load没有s,这样把整个json中的内容key-value的方式存下啦
            temp = open(temp_path, "r")
            temp = json.load(temp)
            #print(temp["objects"]["rectObject"])
            #顺序提取里面的内容即可
            rectObject = temp["objects"]["rectObject"]

            for i in range(len(rectObject)):
            #找到所有不相同的类别名称
                #if rectObject[i]["class"] == "w":
                 #   print("w is in :", in_file)
                if rectObject[i]["class"] in classes:
                    pass
                else:
                    classes.append(rectObject[i]["class"])
            #print(rectObject[0]['class'])
    #返回所有类别名的列表
    return classes
#该函数用于把标注的图像坐标系位置转换成voc方式，即相对整个图像wh的比例
def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

#该函数用于把json中的格式写成txt的voc格式
def convert_annotation(P_Json):
#路径整体有点蹩脚，再优化。目前生成的txt和之前的json在一个文件夹
    out_file = open(opt.source+"/"+"voc" + "/" + P_Json.split("/")[-1][:-4] + "txt", 'w')
    temp = open(P_Json, "r")
    temp = json.load(temp)
    w = int(temp["size"]["width"])
    h = int(temp["size"]["height"])
    rectObject = temp["objects"]["rectObject"]
    print(all_class)
    for i in range(len(rectObject)):
        cls = rectObject[i]["class"]
        cls_id = all_class.index(cls)
        b = (float(rectObject[i]["minX"]), float(rectObject[i]["maxX"]),
             float(rectObject[i]["minY"]), float(rectObject[i]["maxY"]))
        bb = convert((w,h), b)
        #写入txt
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

wd = getcwd()

#遍历所有的json文件并生成
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='Annotations', help='folder saved json file')  # file/folder, 0 for webcam
    opt = parser.parse_args()
    all_class = gen_classes(opt.source)
    files= os.listdir(opt.source)
    if not os.path.exists(opt.source+"/"+"voc"):
        os.makedirs(opt.source+"/"+"voc")
    for in_file in files:
        temp_path = os.path.join(opt.source, in_file)
        if not os.path.isdir(temp_path):
            convert_annotation(temp_path)
