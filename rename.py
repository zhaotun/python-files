#!/usr/bin/python

import os

# 将dir文件夹中的filetype类型的文件按newname_head依次命名

dir = r'F:\Paddle\models-develop\PaddleCV\image_classification\data\ILSVRC2012\val\video'
newname_head = "val_video"
filetype = "jpg"

list = os.listdir(dir) #列出文件夹下所有的目录与文件
print(len(list))
for i in range(0,len(list)):
        path = os.path.join(dir,list[i])
        print(path)
        if os.path.isfile(path):
          #你想对文件的操作
          #filetype = path.split(".")[2]
          if path.endswith(filetype):
             newname = newname_head + "_" + str(i) + "." + filetype
             os.rename( path,os.path.join(dir,newname) )  
print(" %d files have been renamed." % len(list)) 




