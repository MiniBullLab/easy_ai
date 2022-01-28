import matplotlib.pyplot as plt
import cv2
import pandas as pd
import numpy as np
from scipy import signal


trajectory_position_file = open("./trajectory.txt", "r")
trajectory_position = trajectory_position_file.readlines()

# plt.figure(1)
# pa = []
# po = []
# for i, position in enumerate(trajectory_position):
#     pos = float(position[:-1].split(" ")[1])
#     pa.append(i)
#     po.append(pos)
#
# plt.plot(pa, po)
# plt.show()

def norm_trajectory(position_value, x1, y1, x2, y2):
    position_value = np.array(position_value)
    x1 = np.array(x1)
    y1 = np.array(y1)
    x2 = np.array(x2)
    y2 = np.array(y2)

    # normalized,选取一个坐标y2进行平滑
    input_norm = position_value / 1440
    # print('input_norm:',len(input_norm))
    # 计算box周长
    x_norm = x1 / 1440
    y_norm = y1 / 1440
    prm = (x2-x1+y2-y1)*2/1022 # 1022 for normalized
    # print('prm:',len(prm))

    # 计算x和y方向速度，及绝对值速度
    v_x =  []
    for i in range(26, len(x_norm)):
        v_x.append(x_norm[i] - x_norm[i-1]) # 帧间速度
    v_x = np.array([v_x[0]] * 26 + v_x) # 前26frame取第1frame的值
    # print('v_x:',len(v_x))

    v_y =  []
    for i in range(26, len(y_norm)):
        v_y.append(y_norm[i] - y_norm[i-1])
    v_y = np.array([v_y[0]] * 26 + v_y)
    # print('v_y:',len(v_y))

    abs_v = np.sqrt(v_x**2 + v_y**2)
    # print('abs_v:', len(abs_v))

    # 计算绝对值加速度，前25和第0,1帧为0
    acc_tmp1 = np.array([0] + [v_x[i]-v_x[i-1] for i in range(1, len(v_x))])
    acc_tmp2 = np.array([0] + [v_y[i]-v_y[i-1] for i in range(1, len(v_y))])
    acc = np.sqrt(acc_tmp1**2 + acc_tmp2**2)
    # print('acc:', len(acc))

    # 速度卷积核
    knl_v_slide = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
    # 平滑x方向速度
    v_x_conv_0 = signal.convolve(v_x, knl_v_slide[::-1]) / sum(knl_v_slide)
    v_x_conv = v_x_conv_0[25:-24] # 去掉补的前25帧，去掉卷积多出来的部分
    v_x_conv[0] = 0 # 0帧赋为0
    # print('v_x_conv:', len(v_x_conv))

    # 平滑y方向速度
    v_y_conv_0 = signal.convolve(v_y, knl_v_slide[::-1]) / sum(knl_v_slide)
    v_y_conv = v_y_conv_0[25:-24]
    v_y_conv[0] = 0
    # print('v_y_conv:', len(v_y_conv))

    # 平滑后绝对值速度
    v_conv = np.sqrt(v_x_conv**2 + v_y_conv**2)
    # print('v_conv:', len(v_conv))

    # 平滑绝对值速度
    abs_v_conv_0 = signal.convolve(abs_v, knl_v_slide[::-1]) / sum(knl_v_slide)
    abs_v_conv = abs_v_conv_0[25:-24]
    abs_v_conv[0] = 0
    # print('abs_v_conv', len(abs_v_conv))

    # 平滑绝对值加速度
    knl_acc = [1,2,3,4,5,6,7,8,9,10]
    acc_conv_0 = signal.convolve(acc[25:], knl_acc[::-1]) / sum(knl_acc)
    acc_conv = acc_conv_0[:-9]
    # print('acc_conv', len(acc_conv))

    # vlo 低频速度指标，越大越明显
    vlo_weight = 200 # 速度放大权重，影响vlo
    vlo_power = 1 # 速度指数常数，影响vlo
    # 平衡像素速度与box周长之间关系
    vlo = np.minimum(v_conv/prm[25:]*vlo_weight, 1)**vlo_power
    vlo[0] = vlo[1]
    # print('vlo:', len(vlo))

    # 可预测性指标，加速度越高，该指标越低
    acc_weight = 200 # 加速度权重
    acc_bias = 0.02 # 加速度偏置
    predict = 1-np.minimum(1,np.maximum(0, acc_conv/prm[25:]-acc_bias)*acc_weight)
    # print('predict:', len(predict))

    # 平滑原曲线slide
    slide_0 = []
    knl_compens = [i*i for i in range(1,26)] # 卷积核
    input_norm_1 = input_norm[25:]
    for i in range(25):
        slide_tmp = signal.convolve(input_norm_1[:i+1], knl_compens[:i+1][::-1])/sum(knl_compens[:i+1])
        slide_0.append(slide_tmp[i])
    slide_1 = signal.convolve(input_norm_1, knl_compens[::-1])/sum(knl_compens[:i+1])
    slide_1[:25] = slide_0
    slide = slide_1[:-24]
    # print('slide:', len(slide))

    # 原曲线
    origin = input_norm[25:]
    # print('origin:', len(origin))

    # slide与origin间积分面积，用于补偿slide间origin间差值
    lag = (origin-slide)*np.minimum(predict, vlo)
    # print('lag:', len(lag))

    # 平滑积分面积
    knl_compens = [i**2 for i in range(1,26)] # 卷积核
    lag_conv_0 = []
    for i in range(25):
        lag_tmp = signal.convolve(lag[:i+1], knl_compens[:i+1][::-1])/sum(knl_compens[:i+1])
        lag_conv_0.append(lag_tmp[i])
    lag_conv_1 = signal.convolve(lag, knl_compens[::-1])/sum(knl_compens)
    lag_conv_1[:25] = lag_conv_0
    lag_conv = lag_conv_1[:-24]
    # print('lag_conv:', len(lag_conv))

    # 补偿后最终曲线
    compens_weight = 1 # 补偿权重
    output = (slide + lag_conv) * compens_weight

    return slide, output
    # print('output:', len(output))
    # frame = [i for i in range(0, len(output))]
    # plt.figure(2)
    # plt.plot(frame, output)
    # plt.plot(frame, origin)
    # plt.plot(frame, slide)
    # plt.show()




# # data = pd.read_excel('./trajectory_v2.xlsx')
# # box left_up and rihgt_down pixel coordinate
# # -25frame -- 651frame
# # 前25帧取第0帧的值
# x1_l = []
# y1_l = []
# x2_l = []
# y2_l = []
# image = np.zeros((1440, 2560, 3))
# for i, position in enumerate(trajectory_position):
#     x1_l.append(int(position[:-1].split(" ")[2]))
#     y1_l.append(int(position[:-1].split(" ")[3]))
#     x2_l.append(int(position[:-1].split(" ")[4]))
#     y2_l.append(int(position[:-1].split(" ")[5]))
#
#     if len(x1_l) > 26:
#
#         x1 = np.array(x1_l)
#         y1 = np.array(y1_l)
#         x2 = np.array(x2_l)
#         y2 = np.array(y2_l)
#
#         x1_norm_slide, x1_norm_output  = norm_trajectory(x1, x1, y1, x2, y2)
#         x2_norm_slide, x2_norm_output  = norm_trajectory(x2, x1, y1, x2, y2)
#         y2_norm_slide, y2_norm_output  = norm_trajectory(y2, x1, y1, x2, y2)
#         x1_norm_slide, x1_norm_output = x1_norm_slide * 1440, x1_norm_output * 1440
#         x2_norm_slide, x2_norm_output = x2_norm_slide * 1440, x2_norm_output * 1440
#         y2_norm_slide, y2_norm_output = y2_norm_slide * 1440, y2_norm_output * 1440
#
#         trajectory_position_current = (int(x1[i] + (x2[i] - x1[i]) / 2), int(y2[i]))
#         trajectory_position_current_n_s = (int(x1_norm_slide[i-25] + (x2_norm_slide[i-25] - x1_norm_slide[i-25]) / 2), int(y2_norm_slide[i-25]))
#         trajectory_position_current_n_o = (int(x1_norm_output[i-25] + (x2_norm_output[i-25] - x1_norm_output[i-25]) / 2), int(y2_norm_output[i-25]))
#         cv2.circle(image, trajectory_position_current, 2, (0, 0, 255), -1)
#
#         cv2.circle(image, trajectory_position_current_n_s, 2, (0, 255, 255), -1)
#
#         # cv2.circle(image, trajectory_position_current_n_o, 2, (255, 0, 255), -1)
#
#         cv2.namedWindow("image", 0)
#         cv2.imshow("image", image)
#         cv2.waitKey(500)
#

# plot origin trajectory
# image = np.zeros((1440, 2560, 3))
# print(len(x1))
# for x_1, y_1, x_2, y_2, x1_n_s, x1_n_o, x2_n_s, x2_n_o, y2_n_s, y2_n_o\
#         in zip(x1[25:], y1[25:], x2[25:], y2[25:], x1_norm_slide, x1_norm_output, x2_norm_slide, x2_norm_output, y2_norm_slide, y2_norm_output):
#     trajectory_position_current = (int(x_1 + (x_2 - x_1) / 2), int(y_2))
#     trajectory_position_current_n_s = (int(x1_n_s + (x2_n_s - x1_n_s) / 2), int(y2_n_s))
#     trajectory_position_current_n_o = (int(x1_n_o + (x2_n_o - x1_n_o) / 2), int(y2_n_o))
#     print(trajectory_position_current, trajectory_position_current_n_s, trajectory_position_current_n_o)
#     cv2.circle(image, trajectory_position_current, 2, (0, 0, 255), -1)

#     cv2.circle(image, trajectory_position_current_n_s, 2, (0, 255, 255), -1)

#     # cv2.circle(image, trajectory_position_current_n_o, 2, (255, 0, 255), -1)

#     cv2.namedWindow("image", 0)
#     cv2.imshow("image", image)
#     cv2.waitKey()