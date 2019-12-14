#!/usr/bin/env python3
# -*- coding:utf8 -*-

import os
import shutil
import cv2

source_path = os.path.abspath(r'C:\Users\wt\Desktop\test')


if not os.path.exists(target_path):
    os.makedirs(target_path)

i=0
if os.path.exists(source_path):
    # root 所指的是当前正在遍历的这个文件夹的本身的地址
    # dirs 是一个 list，内容是该文件夹中所有的目录的名字(不包括子目录)
    # files 同样是 list, 内容是该文件夹中所有的文件(不包括子目录)
    for root, dirs, files in os.walk(source_path):  # walk 遍历当前source_path目录和所有子目录的文件和目录
        for file in files:                          # files 是所有的文件名列表，
            src_file = os.path.join(root, file)
            img = cv2.imread(src_file)
            HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            dst_file = src_file.replace("test","hsv")
            print(dst_file)
            cv2.imwrite(dst_file,HSV)
            i=i+1

print('%d files moved!'%i)
