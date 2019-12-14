#!/usr/bin/env python3
# -*- coding:utf8 -*-

import os
import shutil
import cv2
import random

import Augmentor # should install by 'pip install Augmentor' first


source_path = os.path.abspath(r'F:\Face\LivingDetect\DataSet\IR\IR_Real_Test')


p = Augmentor.Pipeline(source_path)

# 随机旋转
p.rotate(probability=0.6, max_left_rotation=15, max_right_rotation=15) 

# 随机中心裁剪
p.zoom_random(probability=0.8, percentage_area=0.8,randomise_percentage_area=False)

# 随机透视
#p.skew(probability=0.05)

# 随机扭曲
#p.random_distortion(probability=0.1,grid_width=2,grid_height=9,magnitude=5)

# 随机裁剪指定面积
p.crop_random(probability=0.5,percentage_area=0.8,randomise_percentage_area=False)

# 随机裁剪指定面积
#p.crop_centre(probability=0.8,percentage_area=0.7,randomise_percentage_area=False)

# 随机亮度
p.random_brightness(probability=0.8,min_factor=0.75,max_factor=1.25)

# 随机饱和度
p.random_color(probability=0.3,min_factor=0.75,max_factor=1.25)

# 随机对比度
p.random_contrast(probability=0.1,min_factor=0.75,max_factor=1.25)

# 随机擦除指定面积
#p.random_erasing(probability=0.1,rectangle_area=0.2)


# 随机镜像
#p.flip_random(probability=1)

p.sample(1000)