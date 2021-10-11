#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
import cv2


class VideoProcess():

    def __init__(self):
        self.videoPath = ""
        self.videoCapture = None
        self.video_writer = None
        self.image_size = None
        self.fps = 25

    def isVideoFile(self, videoPath):
        if os.path.exists(videoPath):
            return any(videoPath.endswith(extension) for extension in [".avi", ".mp4", ".mov", ".MPG"])
        else:
            return False

    def save_video(self, save_video_path, size=None):
        if size is not None:
            self.image_size = size
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.video_writer = cv2.VideoWriter(save_video_path, fourcc,
                                            self.fps, self.image_size)
        if self.video_writer is not None:
            return True
        else:
            print("video writer init fail!")
            return False

    def openVideo(self, videoPath):
        self.videoCapture = cv2.VideoCapture(videoPath)
        if self.videoCapture is not None:
            self.fps = self.videoCapture.get(cv2.CAP_PROP_FPS)
            width = int(self.videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.image_size = (width, height)
            return True
        else:
            return False

    def read_frame(self):
        success = False
        frame = None
        if self.videoCapture is not None:
            success, frame = self.videoCapture.read()
        return success, frame

    def write_frame(self, frame):
        self.video_writer.write(frame)

    def readRGBFrame(self):
        success = False
        frame = None
        rgbImage = None
        if self.videoCapture is not None:
            success, frame = self.videoCapture.read()
        if success == True:
            rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return success, frame, rgbImage

    def getFrameCount(self):
        result = 0
        if self.videoCapture is not None:
            result = self.videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
        return result
