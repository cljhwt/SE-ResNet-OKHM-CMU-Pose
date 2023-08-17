"""
Provides different utilities to preprocess images.
Args:
image: A np.array representing an image of (h,w,3).

Returns:
A preprocessed image. which dtype is np.float32
and transposed to (3,h,w).

"""

import cv2
import numpy as np


def rtpose_preprocess(image):
    image = image.astype(np.float32)
    image = image / 256. - 0.5#将数据化为-0.5->+0.5间的数
    # H*W*C->C*H*W(即，H索引为[0],W索引为[1],C索引为[2],即，[0]*[1]*[2]->[2]*[0]*[1])
    image = image.transpose((2, 0, 1)).astype(np.float32)

    return image


def vgg_preprocess(image):
    image = image.astype(np.float32) / 255.
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = image.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]

    preprocessed_img = preprocessed_img.transpose((2, 0, 1)).astype(np.float32)
    return preprocessed_img


def inception_preprocess(image):
    image = image.copy()[:, :, ::-1]
    image = image.astype(np.float32)
    image = image / 128. - 1.
    image = image.transpose((2, 0, 1)).astype(np.float32)

    return image


def ssd_preprocess(image):
    image = image.astype(np.float32)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_image -= (104.0, 117.0, 123.0)

    processed_img = rgb_image.astype(np.float32)
    processed_img = processed_img[:, :, ::-1].copy()
    processed_img = processed_img.transpose((2, 0, 1)).astype(np.float32)

    return processed_img
