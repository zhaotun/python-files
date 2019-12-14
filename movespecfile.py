#!/usr/bin/env python3
# -*- coding:utf8 -*-

import os
import shutil

source_path = os.path.abspath(r'F:\binlang\srcImgs\small\test')
target_path = os.path.abspath(r'F:\binlang\srcImgs\small\test_png')

if not os.path.exists(target_path):
    os.makedirs(target_path)


i=0
if os.path.exists(source_path):
    # root 所指的是当前正在遍历的这个文件夹的本身的地址
    # dirs 是一个 list，内容是该文件夹中所有的目录的名字(不包括子目录)
    # files 同样是 list, 内容是该文件夹中所有的文件(不包括子目录)
    for root, dirs, files in os.walk(source_path):  # walk 遍历当前source_path目录和所有子目录的文件和目录
        for file in files:                          # files 是所有的文件名列表，
            if file.endswith('_jpeg.png') or file.endswith('_jpeg_scoremap.png'):
              src_file = os.path.join(root, file)
              shutil.move(src_file, target_path)
              print(src_file)
              i=i+1

print('%d files moved!'%i)
