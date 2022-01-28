import cv2
from collections import deque


class DataInfo():

    def __init__(self, image):
        super().__init__()
        self.image = image
        self.width = self.image.shape[1]
        self.height = self.image.shape[0]
        self.channel = self.image.shape[2]

class DetResult():

    def __init__(self):
        super().__init__()
        self.current_frame = -1
        self.class_id = -1
        self.track_id = -1
        self.confidence = 0.0
        self.head_location = []
        self.pedestrian_location = []


class DetResultInfo(DataInfo):

    def __init__(self, image):
        super().__init__(image)
        self.det_results_vector = []


class TrajectoryParams():

    def __init__(self):
        super().__init__()
        self.latest_frame_id = -1
        self.draw_flag = 0

        self.pedestrian_direction = -1

        # Track param
        self.relative_distance = 0.0
        self.mean_velocity = 0.0
        self.confidence = 0.0
        self.class_id = -1
        self.velocity_vector = []
        self.head_location = []
        self.pedestrian_location = []
        self.pedestrian_x_start = []
        self.pedestrian_y_start = []
        self.pedestrian_x_end = []
        self.pedestrian_y_end = []
        self.trajectory_position = deque(maxlen=30)
        self.trajectory_bird_position = deque(maxlen=60)