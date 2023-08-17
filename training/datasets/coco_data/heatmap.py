# coding=utf-8

import random
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import misc, ndimage


"""Implement the generate of every channel of ground truth heatmap.
:param centerA: int with shape (2,), every coordinate of person's keypoint.
:param accumulate_confid_map: one channel of heatmap, which is accumulated, 
       np.log(100) is the max value of heatmap.
:param params_transform: store the value of stride and crop_szie_y, crop_size_x                 
"""


def putGaussianMaps(center, accumulate_confid_map, params_transform):
    crop_size_y = params_transform['crop_size_y']
    crop_size_x = params_transform['crop_size_x']
    stride = params_transform['stride']
    sigma = params_transform['sigma']

    grid_y = crop_size_y / stride
    grid_x = crop_size_x / stride
    start = stride / 2.0 - 0.5
    y_range = [i for i in range(int(grid_y))]
    x_range = [i for i in range(int(grid_x))]
    xx, yy = np.meshgrid(x_range, y_range)
    xx = xx * stride + start
    yy = yy * stride + start
    d2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
    exponent = d2 / 2.0 / sigma / sigma
    #mask = exponent <= 4.6052 #sigma=7,半径为20像素的范围
    if int(sigma) == 7:
        mask = exponent <= 4.6052 #4.6052的含义，根号下（2*4.6052）为3,即当d2为3倍的sigma时小于4.6052.而3倍sigma代表事件概率小于千分之三为不可能事件
        #print('sigma = 7')
    if int(sigma) == 15:
        mask = exponent <= 21.1463
        #print('sigma = 15')
    if int(sigma) == 11:
        mask = exponent <= 11.3720
        #print('sigma = 11')
    if int(sigma) == 3:
        mask = exponent <= 0.8458
        #print('sigma = 3')
    cofid_map = np.exp(-exponent)
    cofid_map = np.multiply(mask, cofid_map)
    # 按原文中取对应点概率的最大值
    for i in range(accumulate_confid_map.shape[0]):
        for j in range(accumulate_confid_map.shape[1]):
            if accumulate_confid_map[i][j] < cofid_map[i][j]:
                accumulate_confid_map[i][j] = cofid_map[i][j]
    # accumulate_confid_map += cofid_map
    # accumulate_confid_map[accumulate_confid_map > 1.0] = 1.0
    return accumulate_confid_map
