# -*- coding: utf-8 -*-
import os


#文件目录
root_data_path = r"F:\binlang\TestImage_1111\TestImage\test"

#输出的TXT文件路径
file_path = root_data_path + "\\" + "filepath.txt"

#如果存在之前生成的TXT文件，先把它们删除
if os.path.exists(file_path):
    os.remove(file_path)

#打开要存储的TXT文件
filew = open(file_path, "w")

i=0
for root, dirs, files in os.walk(root_data_path):  # walk 遍历当前source_path目录和所有子目录的文件和目录
   for file in files:                          # files 是所有的文件名列表，
       src_file = os.path.join(root, file)
       print(src_file)
       filew.write(src_file+"\n")
       i=i+1

print('There are %d files !'%i)

filew.close()
