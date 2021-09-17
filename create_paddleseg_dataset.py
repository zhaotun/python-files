# -*- coding: utf-8 -*-
import os
import random

#文件目录
root_data_path = r'D:\tiff\data\croped_images\road_seg_dataset\JPEGImages'

#输出的TXT文件路径
train_list_path = r'D:\tiff\data\croped_images\road_seg_dataset' +  "/train_list.txt"
val_list_path   = r'D:\tiff\data\croped_images\road_seg_dataset' +  "/val_list.txt"
test_list_path  = r'D:\tiff\data\croped_images\road_seg_dataset' +  "/test_list.txt"

#如果存在之前生成的TXT文件，先把它们删除
if os.path.exists(train_list_path):
    os.remove(train_list_path)
if os.path.exists(val_list_path):
    os.remove(val_list_path)
if os.path.exists(test_list_path):
    os.remove(test_list_path)
    
    
#打开要存储的TXT文件
train_list_file = open(train_list_path, "w")
val_list_file = open(val_list_path, "w")
test_list_file = open(test_list_path, "w")

train_ratio = 0.8
val_ratio   = 0.1
test_ratio  = 0.1

i=0
for root, dirs, files in os.walk(root_data_path):  # walk 遍历当前source_path目录和所有子目录的文件和目录
   print('len(files) = ',len(files))
   print('type(files) = ',type(files))
   
   random.shuffle(files)
   
   train_num = int( len(files)* train_ratio )
   val_num   = int( len(files)* val_ratio )
   test_num  = len(files) - train_num - val_num
   
   print('train_num = ',train_num)
   print('val_num = ',val_num)
   print('test_num = ',test_num)
   
   train_list = files[ : train_num]
   val_list   = files[ train_num : train_num + val_num ]
   test_list  = files[ train_num + val_num : ]
   
   print('len(train_list) = ',len(train_list))
   print('len(val_list)   = ',len(val_list))
   print('len(test_list)  = ',len(test_list))
   
   # 写入txt
   for file in train_list: 
       train_list_file.write('JPEGImages/' + file + ' Annotations/' + file.replace('jpg','png') + "\n")
   for file in val_list:                        
       val_list_file.write('JPEGImages/' + file + ' Annotations/' + file.replace('jpg','png') + "\n") 
   for file in test_list:                         
       test_list_file.write('JPEGImages/' + file + ' Annotations/' + file.replace('jpg','png') + "\n")

train_list_file.close()
val_list_file.close()
test_list_file.close()

print('over')