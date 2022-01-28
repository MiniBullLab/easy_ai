import cv2
import time
import numpy as np
from .data_struct import TrajectoryParams
from .track_norm import norm_trajectory


class CalculateTraj():

    def __init__(self, camera_calibration_file):
        super().__init__()

        self.camera_calibration_file = camera_calibration_file
        self.track_idx_map = {}
        self.dTs = 1. / 25.  # the interval between frames, assuming that the camera acquisition frame rate is 25 frames
        self.pixel2world_distance = 0.6 / 46. # 3. / 40.  # assume 40 pixels equal 3 meters
        self.bird_draw_x_start = 0
        self.bird_draw_x_end = 1280
        self.bird_direction_line = [30, 270, 510]

    def bird_view_matrix_calculate(self, image):
        # self.image_resized = image
        self.image_resized = cv2.resize(image, (int(image.shape[1] / 2), int(image.shape[0] / 2)), interpolation=cv2.INTER_LINEAR)
        self.c, self.r = self.image_resized.shape[0:2]
        pst2 = np.float32([[180,162],[618,0],[552,540],[682,464]])
        pst1 = np.float32([[0, 0], [self.r, 0], [0, self.c], [self.r, self.c]])
        self.transferI2B = cv2.getPerspectiveTransform(pst1, pst2)
        self.transferI2B = np.array([[ 3.85507158e+00, 1.91293325e+00, -1.97186911e+03],
                             [ 2.42741397e-02, 4.26011151e+00, -5.73160986e+02],
                             [ 4.83356302e-05, 4.25105526e-03, 1.00000000e+00]])

    def projection_on_bird(self, p):
        M = self.transferI2B
        px = (M[0][0] * p[0] + M[0][1] * p[1] + M[0][2]) / (M[2][0] * p[0] + M[2][1] * p[1] + M[2][2])
        py = (M[1][0] * p[0] + M[1][1] * p[1] + M[1][2]) / (M[2][0] * p[0] + M[2][1] * p[1] + M[2][2])
        return (int(px), int(py))

    def bird_view_transform(self):
        self.bird = cv2.warpPerspective(self.image_resized, self.transferI2B, (self.r, self.c))

    def calculate_trajectory(self, det_results):
        if len(det_results) > 0:
            for key in list(self.track_idx_map.keys()):
                if self.track_idx_map[key].latest_frame_id != -1 and \
                        (det_results[0].current_frame - self.track_idx_map[key].latest_frame_id) > 3:
                    del self.track_idx_map[key]
                    continue

        for i in range(0, len(det_results)):
            current_frame = det_results[i].current_frame
            track_id = det_results[i].track_id
            # if track_id = -1 detection has not match the track result, we don't need to put out
            if track_id == -1:
                continue
            confidence = det_results[i].confidence
            class_id = det_results[i].class_id
            head_loc = det_results[i].head_location
            pedestrian_location = det_results[i].pedestrian_location
            x_start = det_results[i].pedestrian_location[0]
            y_start = det_results[i].pedestrian_location[1]
            x_end = det_results[i].pedestrian_location[2]
            y_end = det_results[i].pedestrian_location[3]

            if track_id in self.track_idx_map:
                if len(self.track_idx_map[track_id].pedestrian_x_start) > 26:
                    x1_norm_slide, x1_norm_output = norm_trajectory(self.track_idx_map[track_id].pedestrian_x_start,
                                                                    self.track_idx_map[track_id].pedestrian_x_start,
                                                                    self.track_idx_map[track_id].pedestrian_y_start,
                                                                    self.track_idx_map[track_id].pedestrian_x_end,
                                                                    self.track_idx_map[track_id].pedestrian_y_end)
                    x2_norm_slide, x2_norm_output = norm_trajectory(self.track_idx_map[track_id].pedestrian_x_end,
                                                                    self.track_idx_map[track_id].pedestrian_x_start,
                                                                    self.track_idx_map[track_id].pedestrian_y_start,
                                                                    self.track_idx_map[track_id].pedestrian_x_end,
                                                                    self.track_idx_map[track_id].pedestrian_y_end)
                    y2_norm_slide, y2_norm_output = norm_trajectory(self.track_idx_map[track_id].pedestrian_y_end,
                                                                    self.track_idx_map[track_id].pedestrian_x_start,
                                                                    self.track_idx_map[track_id].pedestrian_y_start,
                                                                    self.track_idx_map[track_id].pedestrian_x_end,
                                                                    self.track_idx_map[track_id].pedestrian_y_end)
                    x1_norm_slide, x1_norm_output = x1_norm_slide * 1440, x1_norm_output * 1440
                    x2_norm_slide, x2_norm_output = x2_norm_slide * 1440, x2_norm_output * 1440
                    y2_norm_slide, y2_norm_output = y2_norm_slide * 1440, y2_norm_output * 1440

                    trajectory_position_current = (int(x1_norm_output[-1] + (x2_norm_output[-1] - x1_norm_output[-1]) / 2), \
                                                   int(y2_norm_output[-1]))
                else:
                    trajectory_position_current = (int(x_start + (x_end - x_start) / 2), \
                                                   int(y_end))
            else:
                trajectory_position_current = (int(x_start + (x_end - x_start) / 2), \
                                               int(y_end))

            # have problem !!!!!
            trajectory_position_bird = self.projection_on_bird((int(trajectory_position_current[0] / 2.), \
                                                               int(trajectory_position_current[1] / 2.)))

            if track_id in self.track_idx_map:
                move_distance = ((self.track_idx_map[track_id].trajectory_bird_position[-1][0] - trajectory_position_bird[0]) \
                                 ** 2 + (self.track_idx_map[track_id].trajectory_bird_position[-1][1] - trajectory_position_bird[1]) \
                                 ** 2) ** 0.5

                if abs(trajectory_position_bird[1] - self.bird_direction_line[0]) < 5 or \
                        abs(trajectory_position_bird[1] - self.bird_direction_line[1]) < 5 or \
                        abs(trajectory_position_bird[1] - self.bird_direction_line[2]) < 5:
                    if self.track_idx_map[track_id].trajectory_bird_position[-1][1] > trajectory_position_bird[1]:
                        trajectory_direction = 1  # backward
                    else:
                        trajectory_direction = 0  # forward
                else:
                    trajectory_direction = self.track_idx_map[track_id].pedestrian_direction

                self.track_idx_map[track_id].trajectory_position.append(trajectory_position_current)
                self.track_idx_map[track_id].trajectory_bird_position.append(trajectory_position_bird)
                self.track_idx_map[track_id].draw_flag = 1
                self.track_idx_map[track_id].latest_frame_id = current_frame
                self.track_idx_map[track_id].pedestrian_direction = trajectory_direction
                self.track_idx_map[track_id].relative_distance = (self.image_resized.shape[0] - trajectory_position_bird[1])\
                                                                 * self.pixel2world_distance

                velocity_current = move_distance * self.pixel2world_distance / self.dTs
                self.track_idx_map[track_id].pedestrian_x_start.append(x_start)
                self.track_idx_map[track_id].pedestrian_y_start.append(y_start)
                self.track_idx_map[track_id].pedestrian_x_end.append(x_end)
                self.track_idx_map[track_id].pedestrian_y_end.append(y_end)
                self.track_idx_map[track_id].pedestrian_location = pedestrian_location
                self.track_idx_map[track_id].head_location = head_loc
                self.track_idx_map[track_id].velocity_vector.append(velocity_current)
                if len(self.track_idx_map[track_id].trajectory_bird_position) < 3:
                    self.track_idx_map[track_id].mean_velocity = 1.38
                else:
                    self.track_idx_map[track_id].mean_velocity = sum(self.track_idx_map[track_id].velocity_vector[-3:]) / 3
                self.track_idx_map[track_id].confidence = confidence
                self.track_idx_map[track_id].class_id = class_id
            else:
                trajector_param = TrajectoryParams()
                trajector_param.draw_flag = 1
                trajector_param.latest_frame_id = current_frame
                trajector_param.trajectory_position.append(trajectory_position_current)
                trajector_param.trajectory_bird_position.append(trajectory_position_bird)
                trajector_param.pedestrian_direction = -1
                trajector_param.relative_distance = (self.image_resized.shape[0] - trajectory_position_bird[1])\
                                                                 * self.pixel2world_distance

                # 前26frame取第1frame的值
                trajector_param.pedestrian_x_start = [x_start] * 26
                trajector_param.pedestrian_y_start = [y_start] * 26
                trajector_param.pedestrian_x_end = [x_end] * 26
                trajector_param.pedestrian_y_end = [y_end] * 26
                trajector_param.pedestrian_location = pedestrian_location
                trajector_param.head_location = head_loc
                trajector_param.velocity_vector.append(1.38)
                trajector_param.mean_velocity = 1.38
                trajector_param.confidence = confidence
                trajector_param.class_id = class_id

                self.track_idx_map[track_id] = trajector_param

    def image_show(self, image, window_name):
        cv2.namedWindow(window_name, 0)
        # cv2.resizeWindow(window_name, (int(image.shape[1]*0.7), int(image[0]*0.7)))
        cv2.imshow(window_name, image)
        key = cv2.waitKey()

    def trajectory_show(self):
        color = (0, 128, 128)
        txt_color = (255, 255, 255)
        bird_image = np.zeros((self.image_resized.shape[0], self.bird_draw_x_end - self.bird_draw_x_start, 3))
        crowded_counted = len(self.track_idx_map)
        system_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        for key, value in self.track_idx_map.items():
            # p1, p2 = (int(value.pedestrian_x_start[-1] / 2.), int(value.pedestrian_y_start[-1] / 2.)), \
            #          (int(value.pedestrian_x_end[-1] / 2.), int(value.pedestrian_y_end[-1] / 2.))
            # p1, p2 = (int(value.pedestrian_location[0] / 2.), int(value.pedestrian_location[1] / 2.)), \
            #          (int(value.pedestrian_location[2] / 2.), int(value.pedestrian_location[3] / 2.))
            p1, p2 = (int(value.pedestrian_location[0]), int(value.pedestrian_location[1])), \
                     (int(value.pedestrian_location[2]), int(value.pedestrian_location[3]))
            # p1_head, p2_head = (value.head_location[0], value.head_location[1]), \
            #                    (value.head_location[2], value.head_location[3])
            cv2.rectangle(self.image_resized, p1, p2, color, thickness=1, lineType=cv2.LINE_AA)
            # cv2.rectangle(self.image_resized, p1_head, p2_head, (255, 255, 0), thickness=1, lineType=cv2.LINE_AA)
            label = f'{key} {value.pedestrian_direction} {value.mean_velocity:.2f} {value.relative_distance:.2f}'

            # label = f'{key} Ped {value.confidence:.2f}'
            tf = max(2 - 1, 1)  # font thickness
            w, h = cv2.getTextSize(label, 0, fontScale=1 / 3, thickness=tf)[0]  # text width, height
            outside = p1[1] - h - 3 >= 0  # label fits outside box
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(self.image_resized, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(self.image_resized, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, 1 / 3, txt_color,
                        thickness=tf, lineType=cv2.LINE_AA)
            for point in value.trajectory_position:
                # cv2.circle(self.image_resized, (int(point[0] / 2.), int(point[1] / 2.)), 2, (0, 0, 255), -1)
                cv2.circle(self.image_resized, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)
            cv2.putText(self.image_resized, f"Counted: {crowded_counted}", (20, 60), 0, 2 / 3, (255, 255, 255),
                        thickness=tf, lineType=cv2.LINE_AA)
            cv2.putText(self.image_resized, f"{system_time}", (20, 30), 0, 2 / 3, (255, 255, 255),
                        thickness=tf, lineType=cv2.LINE_AA)

            # bird image show
            p1_bird, p2_bird = (int(value.trajectory_bird_position[-1][0] - self.bird_draw_x_start - 5), \
                                int(value.trajectory_bird_position[-1][1] - 5)), \
                               (int(value.trajectory_bird_position[-1][0] - self.bird_draw_x_start + 5), \
                                int(value.trajectory_bird_position[-1][1] + 5))
            cv2.rectangle(bird_image, p1_bird, p2_bird, color, thickness=1, lineType=cv2.LINE_AA)
            cv2.putText(bird_image, label, (p1_bird[0], p1_bird[1] - 2 if outside else p1_bird[1] + h + 2), 0, 1 / 4, txt_color,
                        thickness=tf, lineType=cv2.LINE_AA)
            if value.pedestrian_direction == 0:
                cv2.line(bird_image, (int(p2_bird[0] - 5), int(p2_bird[1] - 2)), (int(p2_bird[0] - 5), \
                                                                                    int(p2_bird[1] + 2)), (0, 255, 0), 2)
            elif value.pedestrian_direction == 1:
                cv2.line(bird_image, (int(p1_bird[0] + 5), int(p1_bird[1] - 2)), (int(p1_bird[0] + 5), \
                                                                                int(p1_bird[1] + 2)), (0, 255, 0), 2)
            for line_height_point in self.bird_direction_line:
                cv2.line(bird_image, (0, line_height_point), (bird_image.shape[1], line_height_point), (255, 0, 0), 1)
            for bird_point in value.trajectory_bird_position:
                cv2.circle(bird_image, ((bird_point[0] - self.bird_draw_x_start), bird_point[1]), 1, (0, 0, 255), -1)
            cv2.putText(bird_image, f"Counted: {crowded_counted}", (10, 30), 0, 1 / 3, (255, 255, 0),
                        thickness=tf, lineType=cv2.LINE_AA)
            cv2.putText(bird_image, f"{system_time}", (10, 15), 0, 1 / 3, (255, 255, 0),
                        thickness=tf, lineType=cv2.LINE_AA)

        self.image_show(self.image_resized, "traj_show")
        self.image_show(bird_image, "bird_map_show")
        self.image_show(self.bird, "bird_show")