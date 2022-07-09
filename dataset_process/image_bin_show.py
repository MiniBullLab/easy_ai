import os
import cv2
import numpy as np
from easyai.helper.dir_process import DirProcess


def read_feature_map(feature_path, image_size):
    feature_map = np.fromfile(feature_path, dtype=np.uint8)
    result = feature_map.reshape(image_size[1], image_size[0])
    return result


def main():
    dir_path = "/home/lpj/dataset/luyanshiluzhi/48/tof"
    dir_process = DirProcess()
    list_path = list(dir_process.getDirFiles(dir_path, "*.bin"))
    data_list = sorted(list_path)
    for index, bin_path in enumerate(data_list):
        path, file_name_and_post = os.path.split(bin_path)
        tof_path = os.path.join(path, "tof_%d.bin" % index)
        image_path = os.path.join(path, "tof_%d.jpg" % index)
        print(tof_path)
        image = read_feature_map(tof_path, (240, 180))
        mask = image != 0
        image[mask] = 255 - image[mask]
        image.tofile(tof_path)
        # cv2.imshow("image", image)
        cv2.imwrite(image_path, image)
        # if cv2.waitKey(0) & 0xff == ord('q'):
        #     break


def main1():
    dir_path = "/home/lpj/dataset/luyanshiluzhi/2/tof"
    dir_process = DirProcess()
    list_path = list(dir_process.getDirFiles(dir_path, "*.bin"))
    data_list = sorted(list_path)
    for index, bin_path in enumerate(data_list):
        path, file_name_and_post = os.path.split(bin_path)
        tof_path = os.path.join(path, "tof_%d.bin" % index)
        print(tof_path)
        image = read_feature_map(tof_path, (240, 180))
        cv2.imshow("image", image)
        if cv2.waitKey(0) & 0xff == ord('q'):
            break


if __name__ == "__main__":
   main()
