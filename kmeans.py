# -*- coding: utf-8 -*-
# @Author  : matthew
# @Software: PyCharm

import cv2
import matplotlib.pyplot as plt
import numpy as np

def seg_kmeans_color():
    img = cv2.imread('1.bmp', cv2.IMREAD_COLOR)
    # 变换一下图像通道bgr->rgb，否则很别扭啊
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])

    # 3个通道展平
    img_flat = img.reshape((img.shape[0] * img.shape[1], 3))
    img_flat = np.float32(img_flat)

    # 迭代参数
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 20, 0.5)
    flags = cv2.KMEANS_RANDOM_CENTERS

    # 聚类
    compactness, labels, centers = cv2.kmeans(img_flat, 4, None, criteria, 10, flags)

    # 显示结果
    img_output = labels.reshape((img.shape[0], img.shape[1]))
    plt.subplot(121), plt.imshow(img), plt.title('input')
    plt.subplot(122), plt.imshow(img_output, 'gray'), plt.title('kmeans')
    plt.show()


if __name__ == '__main__':
    seg_kmeans_color()