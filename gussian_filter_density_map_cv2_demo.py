"""

利用高斯模糊生成密度图，并进行可视化展示

1、numpy创建空白图；
2、对空白图相应位置赋值，得到gt图；
3、对gt图进行高斯模糊，得到密度图，float64格式的
4、对密度图进行归一化，转为 uint8 格式，便于 opencv 处理
5、进行伪彩色变换

可以用来对车流密度进行可视化
"""
import numpy as np
import cv2
import scipy.ndimage

if __name__ == '__main__':
    
    # 利用numpy创建空白图片
    gt = np.zeros((540,960))
    
    # 给空白图片相应位置赋值
    gt[11][22]   += 1
    gt[111][111] += 1
    gt[222][222] += 1
    gt[66][66]   += 1
    gt[123][123] += 1
    gt[321][321] += 1
    gt[88][88]   += 1

    for j in range(270,280,1):
        for i in range(510,550,1):
           gt[j][i] += 1
       
    for j in range(200,210,1):
        for i in range(510,530,1):
           gt[j][i] += 1
       

    # 利用numpy创建空白图片，做为密度图的初始化
    density = np.zeros(gt.shape)

    # 对 gt 图像进行高斯模糊得到密度图
    sigma = 16
    density += scipy.ndimage.filters.gaussian_filter(gt, sigma, mode='constant')

    # 对密度图进行归一化，将 float64格式的 np.array 转换为 uint8 格式，否则无法用opencv进行处理
    img_n = cv2.normalize(src=density, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imshow("img_n", img_n)
    
    # 将归一化的密度图转换为伪彩色图
    for i in range(0,13):
        density1_color = cv2.applyColorMap(img_n, i)
        cv2.imshow('d1_color',density1_color)
        cv2.waitKey(0)
        cv2.imwrite( str(i) + '.jpg', density1_color)
