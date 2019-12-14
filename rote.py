#!/usr/bin/env python3
# -*- coding:utf8 -*-

import os
import shutil

import numpy as np
import cv2
import argparse

rote = -90

source_path = os.path.abspath(r'C:\Users\wt\Camera\VID_20191017_170041 (2019-10-18 13-45-24)')


# 定义旋转rotate函数
def rotate(image, angle, center=None, scale=1.0):
    # 获取图像尺寸
    (h, w) = image.shape[:2]
 
    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)
 
    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
 
    # 返回旋转后的图像
    return rotated

i=0
if os.path.exists(source_path):
    # root 所指的是当前正在遍历的这个文件夹的本身的地址
    # dirs 是一个 list，内容是该文件夹中所有的目录的名字(不包括子目录)
    # files 同样是 list, 内容是该文件夹中所有的文件(不包括子目录)
    for root, dirs, files in os.walk(source_path):  # walk 遍历当前source_path目录和所有子目录的文件和目录
        for file in files:                          # files 是所有的文件名列表，
            src_file = os.path.join(root, file)
            
            img = cv2.imread(src_file)
            rotated = rotate(img,rote)
            #img90 = np.rot90(img)
            cv2.imwrite(src_file,rotated)
            #shutil.copy(src_file, target_path)
            print(src_file)
            i=i+1

print('%d files moved!'%i)
