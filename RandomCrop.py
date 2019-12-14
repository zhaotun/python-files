#!/usr/bin/env python3
# -*- coding:utf8 -*-

import os
import shutil
import cv2
import random

source_path = os.path.abspath(r'F:\Face\DataSet\face_anti_spoofing\RGB\RGB_Print_Faces\print\test')
target_path = os.path.abspath(r'C:\Users\wt\Desktop\test')

i=0

'''
随机裁剪
area_ratio为裁剪画面占原画面的比例
hw_vari是扰动占原高宽比的比例范围
'''
def random_crop(img_path, area_ratio, hw_vari):

    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    hw_delta = np.random.uniform(-hw_vari, hw_vari)
    hw_mult = 1 + hw_delta
	
	# 下标进行裁剪，宽高必须是正整数
    w_crop = int(round(w*np.sqrt(area_ratio*hw_mult)))
	
	# 裁剪宽度不可超过原图可裁剪宽度
    if w_crop > w:
        w_crop = w
		
    h_crop = int(round(h*np.sqrt(area_ratio/hw_mult)))
    if h_crop > h:
        h_crop = h
	
	# 随机生成左上角的位置
    x0 = np.random.randint(0, w-w_crop+1)
    y0 = np.random.randint(0, h-h_crop+1)
	
    return crop_image(img, x0, y0, w_crop, h_crop)


def crop(img_path):
   img = cv2.imread(img_path)
   h,w,c = img.shape
   
   list = img_path.split("\\")
   img_name = list[ len(list)-1 ][:-4]
   
   for i in range(0,5):
     x1 = (int) ( random.uniform(0,w * 0.5) )
     y1 = (int) ( random.uniform(0,h * 0.5) )
     x2 = (int) ( x1 + w * 0.5 )
     y2 = (int) ( y1 + h * 0.5 )

     crop_img = img[y1:y2,x1:x2]

     crop_img_name = img_name + "_" + str(i) + ".jpg"
     crop_img_path = os.path.join(target_path, crop_img_name)
     cv2.imwrite(crop_img_path,crop_img)

   
   
    # root 所指的是当前正在遍历的这个文件夹的本身的地址
    # dirs 是一个 list，内容是该文件夹中所有的目录的名字(不包括子目录)
    # files 同样是 list, 内容是该文件夹中所有的文件(不包括子目录)
for root, dirs, files in os.walk(source_path):  # walk 遍历当前source_path目录和所有子目录的文件和目录
   #print(root)
   #print(dirs)
   for file in files:                          # files 是所有的文件名列表，
       src_file = os.path.join(root, file)
       print(src_file)
       i=i+1
       crop(src_file)

print('There are %d files !'%i)
