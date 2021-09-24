import os
import cv2
import numpy as np
from easyai.helper.dir_process import DirProcess


def write_data(image_path, text, save_file):
    path, file_name_post = os.path.split(image_path)
    write_content = "%s|%s\n" % (file_name_post, text)
    save_file.write(write_content)


def txt_process(text_path):
    path, _ = os.path.split(text_path)
    images_dir = os.path.join(path, "../JPEGImages")
    save_train_path = os.path.join(path, "train.txt")
    save_val_path = os.path.join(path, "val.txt")
    dir_process = DirProcess()
    save_train_file = open(save_train_path, "w", encoding='utf-8')
    save_test_file = open(save_val_path, "w", encoding='utf-8')
    image_index = 0
    for line_data in dir_process.getFileData(text_path):
        data_list = [x.strip() for x in line_data.split('|', 1) if x.strip()]
        if len(data_list) == 2:
            image_path = os.path.join(images_dir, data_list[0])
            # print(image_path)
            if os.path.exists(image_path):
                temp_text = data_list[1].strip('\"')
                # print(image_path, temp_text)
                image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), 1)
                dst_img_height, dst_img_width = image.shape[0:2]
                if dst_img_height * 1.0 / dst_img_width >= 1.5:
                    image = np.rot90(image)
                    path, image_name = os.path.split(image_path)
                    cv2.imwrite(image_name, image)
                    print(image_path, "90")
                    if (image_index + 1) % 4 == 0:
                        write_data(image_path, temp_text, save_test_file)
                    else:
                        write_data(image_path, temp_text, save_train_file)
                    image_index += 1
            else:
                print("%s not exist" % image_path)
        else:
            print("% error" % line_data)
    save_train_file.close()
    save_test_file.close()


def image_dir_process(image_dir):
    save_train_path = os.path.join(image_dir, "../train.txt")
    save_val_path = os.path.join(image_dir, "../val.txt")
    dir_process = DirProcess()
    save_train_file = open(save_train_path, "w", encoding='utf-8')
    save_test_file = open(save_val_path, "w", encoding='utf-8')
    image_index = 0
    for image_path in dir_process.getDirFiles(image_dir, "*.jpg"):
        path, image_name = os.path.split(image_path)
        # label_text = image_name.split("_", 1)[0]
        label_text = image_name.split(".", 1)[0]
        if os.path.exists(image_path):
            re_image_path = os.path.join(path, "img_%05d.jpg" % image_index)
            os.rename(image_path, re_image_path)
            image = cv2.imdecode(np.fromfile(re_image_path, dtype=np.uint8), 1)
            if (image_index + 1) % 10 == 0:
                write_data(re_image_path, label_text.strip(), save_test_file)
            else:
                write_data(re_image_path, label_text.strip(), save_train_file)
            image_index += 1


if __name__ == "__main__":
    # txt_process("/home/lpj/dataset/ocr/Synthetic_text/ImageSets/gt.txt")
    image_dir_process("/home/lpj/dataset/ocr/lpr_test/JPEGImages")
