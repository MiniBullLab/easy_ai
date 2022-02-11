import numpy as np
import torch

from .nn_matching import NearestNeighborDistanceMetric
from .detection import Detection
from .tracker import Tracker

from easyai.helper import TrackObject2d


__all__ = ['DeepSort']


class DeepSort():

    def __init__(self, max_dist=0.2, min_confidence=0.3,
                 max_iou_distance=0.7, max_age=70,
                 n_init=3, nn_budget=100):
        self.image_size = None  # (width, height)
        self.min_confidence = min_confidence
        max_cosine_distance = max_dist
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric, max_iou_distance=max_iou_distance,
                               max_age=max_age, n_init=n_init)

    def process(self, reid_objects, image_size):
        result = []
        self.image_size = image_size
        xywhs = np.zeros((len(reid_objects), 4), dtype=np.float32)
        confs = np.zeros(len(reid_objects))
        clss = np.zeros(len(reid_objects))
        features = []
        for index, temp_object in enumerate(reid_objects):
            x, y = temp_object.center()
            width = temp_object.width()
            height = temp_object.height()
            xywhs[index, :] = np.array([x, y, width, height])
            confs[index] = temp_object.classConfidence
            clss[index] = temp_object.classIndex
            features.append(temp_object.reid)
        features = np.array(features)
        outputs = self.update(xywhs, confs, clss, features)
        for output in outputs:
            temp = TrackObject2d()
            temp.track_id = int(output[4])
            temp.min_corner.x = output[0]
            temp.min_corner.y = output[1]
            temp.max_corner.x = output[2]
            temp.max_corner.y = output[3]
            temp.classIndex = int(output[5])
            temp.classConfidence = output[6]
            result.append(temp)
        return result

    def update(self, bbox_xywh, confidences, classes, features):
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(
            confidences) if conf > self.min_confidence]

        # print(len(detections))
        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections, classes, confidences)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            class_id = track.class_id
            confidence = track.confidence
            outputs.append(np.array([x1, y1, x2, y2, track_id, class_id, confidence])) # , dtype=np.int
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs

    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """
    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x+w), self.image_size[0] - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y+h), self.image_size[1] - 1)
        return x1, y1, x2, y2

    def increment_ages(self):
        self.tracker.increment_ages()

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy
        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h

