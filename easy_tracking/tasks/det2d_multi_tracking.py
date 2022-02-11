#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.utility.logger import EasyLogger
if EasyLogger.check_init():
    log_file_path = EasyLogger.get_log_file_path("tracking_task.log")
    EasyLogger.init(logfile_level="debug", log_file=log_file_path, stdout_level="error")
import cv2
from pathlib import Path
from easy_tracking.reid.det2d_deep_sort import Det2dDeepSort
from easy_tracking.deep_sort.deep_sort import DeepSort
from easy_tracking.reid.det2d_fairmot import Det2dFairMOT
from easy_tracking.fairmot.jde_tracker import JDETracker
from easyai.helper import DirProcess
from easyai.helper import VideoProcess
from easyai.helper.arguments_parse import TaskArgumentsParse


class MultiDet2dTracking():

    def __init__(self, model_name, gpu_id, weight_path, config_path):
        self.dir_process = DirProcess()
        self.video_process = VideoProcess()
        # self.reid_task = Det2dDeepSort(model_name, gpu_id, weight_path, config_path)
        # self.multi_tracker = DeepSort()
        self.reid_task = Det2dFairMOT(model_name, gpu_id, weight_path, config_path)
        self.multi_tracker = JDETracker()

    def process(self, input_path, is_show=False):
        if Path(input_path).is_dir():
            list_path = list(self.dir_process.getDirFiles(input_path, "*.*"))
            data_list = self.dir_process.sort_path(list_path)
            for image_path in data_list:
                print(image_path)
                cv_image = cv2.imread(image_path)  # BGR
                if cv_image is None:
                    continue
                result = self.single_image_process(cv_image)
                if is_show:
                    if not self.show_result(cv_image, result):
                        break
        elif self.video_process.isVideoFile(input_path):
            if self.video_process.openVideo(input_path):
                while True:
                    success, cv_image = self.video_process.read_frame()
                    if not success:
                        break
                    result = self.single_image_process(cv_image)
                    if is_show:
                        if not self.show_result(cv_image, result):
                            break
            else:
                print("%s: open video fail!" % input_path)
        else:
            print("input error:", input_path)

    def single_image_process(self, src_image):
        image_size = (src_image.shape[1], src_image.shape[0])  # [width, height]
        reid_objects = self.reid_task.process(src_image)
        outputs = self.multi_tracker.process(reid_objects,
                                             image_size)
        return outputs

    def show_result(self, src_image, result):
        for temp in result:
            p1, p2 = (int(temp.min_corner.x), int(temp.min_corner.y)), \
                     (int(temp.max_corner.x), int(temp.max_corner.y))
            cv2.rectangle(src_image, p1, p2, (0, 255, 255), thickness=2, lineType=cv2.LINE_AA)
            label = f'{temp.track_id} {temp.classIndex} {temp.classConfidence:.2f}'
            w, h = cv2.getTextSize(label, 0, fontScale=1 / 3, thickness=1)[0]  # text width, height
            outside = p1[1] - h - 3 >= 0  # label fits outside box
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(src_image, p1, p2, (0, 255, 255), -1, cv2.LINE_AA)  # filled
            cv2.putText(src_image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, 1 / 3, (0, 0, 0),
                        thickness=1, lineType=cv2.LINE_AA)
        # cv2.namedWindow("image", 0)
        cv2.imshow("image", src_image)
        if cv2.waitKey(0) & 0xff == ord('q'):  # 按q退出
            cv2.destroyAllWindows()
            return False
        else:
            return True


def main():
    EasyLogger.info("tracking process start...")
    options = TaskArgumentsParse.inference_parse_arguments()
    multi_tracker_task = MultiDet2dTracking(options.model, 0, options.weights, options.config_path)
    multi_tracker_task.process(options.inputPath, options.show)
    EasyLogger.info("tracking process end!")


if __name__ == '__main__':
    main()
