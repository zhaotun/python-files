#!/usr/bin/env python3
# -*- coding:utf8 -*-

import os
import shutil

source_path = os.path.abspath(r'F:\Face\DataSet\CASIA-FASD\CASIA-FASD-Imgs\Train\Real-imgs')
target_path = os.path.abspath(r'F:\Face\DataSet\face_anti_spoofing\IR\IR_video\IR_Print_faces')

#if not os.path.exists(target_path):
#    os.makedirs(target_path)

i=0

    # root 所指的是当前正在遍历的这个文件夹的本身的地址
    # dirs 是一个 list，内容是该文件夹中所有的目录的名字(不包括子目录)
    # files 同样是 list, 内容是该文件夹中所有的文件(不包括子目录)
for root, dirs, files in os.walk(source_path):  # walk 遍历当前source_path目录和所有子目录的文件和目录
   #print(root)
   #print(dirs)
   for file in files:                          # files 是所有的文件名列表，
       src_file = os.path.join(root, file)
       print(file)
       i=i+1

print('There are %d files !'%i)
