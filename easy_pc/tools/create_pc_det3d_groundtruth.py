#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
import pickle
import numpy as np
from easyai.helper import DirProcess
from easyai.helper.json_process import JsonProcess
from easyai.utility.logger import EasyLogger

from easy_pc.helper.pointcloud_process import PointCloudProcess
from easy_pc.dataloader.utility.box3d_op import points_in_rbbox


class CreatePointCloudDet3dGTSample():

    def __init__(self, point_features):
        self.dir_process = DirProcess()
        self.json_process = JsonProcess()
        self.annotation_name = "../Annotations"
        self.pc_dir_name = "../pcds"
        self.annotation_post = ".json"

        self.pointcloud_process = PointCloudProcess(dim=point_features)

    def create_groundtruth_database(self, train_path, detect3d_class):
        path, _ = os.path.split(train_path)
        pcd_dir = os.path.join(path, self.pc_dir_name)
        annotation_dir = os.path.join(path, self.annotation_name)

        root_path = os.path.join(path, "../")
        database_save_path = os.path.join(root_path, 'gt_database')
        db_info_save_path =os.path.join(root_path, "dbinfos_train.pkl")
        if not os.path.exists(database_save_path):
            os.makedirs(database_save_path)
        all_db_infos = {}

        for filename_and_post in self.dir_process.getFileData(train_path):
            filename, post = os.path.splitext(filename_and_post)
            annotation_filename = filename + self.annotation_post
            annotation_path = os.path.join(annotation_dir, annotation_filename)
            pc_path = os.path.join(pcd_dir, filename_and_post)
            # print(pc_path)
            if os.path.exists(annotation_path) and os.path.exists(pc_path):
                point_cloud = self.pointcloud_process.read_pointcloud(pc_path)
                _, box3d_list = self.json_process.parse_rect3d_data(annotation_path)
                num_obj = len(box3d_list)
                point_indices = None
                if num_obj > 0:
                    box3d_locs = []
                    for box3d in box3d_list:
                        box = [box3d.center.x,
                               box3d.center.y,
                               box3d.center.z,
                               box3d.size.x,
                               box3d.size.y,
                               box3d.size.z,
                               box3d.rotation.z]
                        box3d_locs.append(box)
                    gt_boxes = np.array(box3d_locs).astype(np.float32)
                    point_indices = points_in_rbbox(point_cloud, gt_boxes)
                for i in range(num_obj):
                    gt_filename = "{}_{}_{}.bin".format(filename,
                                                        box3d_list[i].name,
                                                        i)
                    gt_path = os.path.join(database_save_path, gt_filename)
                    gt_points = point_cloud[point_indices[:, i]]
                    if gt_points.shape[0] >= 5 and box3d_list[i].name in detect3d_class:
                        gt_points[:, :3] -= gt_boxes[i, :3]
                        with open(gt_path, 'w') as f:
                            gt_points.tofile(f)

                        db_path = os.path.join("gt_database", gt_filename)
                        db_info = {
                            "name": box3d_list[i].name,
                            "path": db_path,
                            "image_idx": filename,
                            "gt_idx": i,
                            "box3d_lidar": gt_boxes[i],
                            "num_points_in_gt": gt_points.shape[0],
                        }
                        if box3d_list[i].name in all_db_infos:
                            all_db_infos[box3d_list[i].name].append(db_info)
                        else:
                            all_db_infos[box3d_list[i].name] = [db_info]
            else:
                EasyLogger.error("%s or %s not exist" % (annotation_path, pc_path))

        if len(all_db_infos) > 0:
            with open(db_info_save_path, 'wb') as f:
                pickle.dump(all_db_infos, f)


def main():
    print("start...")
    test_process = CreatePointCloudDet3dGTSample(4)
    test_process.create_groundtruth_database("", ())
    print("End of game, have a nice day!")


if __name__ == "__main__":
   main()


