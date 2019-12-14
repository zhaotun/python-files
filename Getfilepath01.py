# -*- coding: utf-8 -*-
import os


#文件目录
root_data_path = r"F:\Paddle\models-develop\PaddleCV\image_classification\data\IR-Faces-10181\test"

#输出的TXT文件路径
file_path = root_data_path + "\\" + "val_list.txt"

#如果存在之前生成的TXT文件，先把它们删除
if os.path.exists(file_path):
    os.remove(file_path)

#打开要存储的TXT文件
filew = open(file_path, "w")

i=0
for root, dirs, files in os.walk(root_data_path):  # walk 遍历当前source_path目录和所有子目录的文件和目录
   for file in files:                          # files 是所有的文件名列表，
       #print(root)
       src_file = os.path.join(root, file)
       #print(src_file)
       #if root.endswith("0"):
       #  filew.write(src_file+" "+"0"+"\n")
       #elif root.endswith("1"):
       #  filew.write(src_file+" "+"1"+"\n")
       #elif root.endswith("video"):
       #  filew.write(src_file+" "+"2"+"\n")
 
       if file.startswith("P"):
         filew.write(src_file+" "+"0"+"\n")
       elif file.startswith("R"):
         filew.write(src_file+" "+"1"+"\n")
 
       i=i+1

print('There are %d files !'%i)

filew.close()
