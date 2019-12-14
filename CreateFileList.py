# -*- coding: utf-8 -*-
import os


#数据集根目录所在路径
root_data_path = r"F:\ImageSegmentation\PaddleSeg\dataset\binglang\images"

#输出的TXT文件路径
traintxt_path = root_data_path + "\\" + "train.txt"
valtxt_path   = root_data_path + "\\" + "val.txt"

#如果存在之前生成的TXT文件，先把它们删除
if os.path.exists(traintxt_path):
    os.remove(traintxt_path)
if os.path.exists(valtxt_path):
    os.remove(valtxt_path)
#返回数据集文件夹下的子文件夹
filenames = os.listdir(root_data_path)    #["train", "val"]

#打开要存储的TXT文件
file_train = open(traintxt_path, "w")
file_val = open(valtxt_path, "w")

if len(filenames) > 0:
    for fn in filenames:
        #数据集根目录下的子文件夹路径，train和val的绝对路径
        full_filename = os.path.join(root_data_path, fn)
        #print(full_filename)
        #print(fn)
        # 判断是训练集文件夹，还是验证集文件夹
        if fn == "train":
            #找出训练集文件夹下的子文件夹名字，是每个类别的文件夹,0表示狗，1表示猫
            file = os.listdir(full_filename)   #["0", "1"]
            for name in file:
                #获得train文件夹下的子文件夹“0”和“1”的绝对路径
                temp = os.path.join(full_filename, name)
                for img in os.listdir(temp):    #分别遍历两个文件夹["0", "1"]下的所有图像
                    #将图像的信息写入到train.txt文件
                    file_train.write(root_data_path + "\\"+name + "\\" + img + " " + name + "\n")
        elif fn == "val":        #当进入到val文件夹后
            #找出训练集文件夹下的子文件夹名字，是每个类别的文件夹,0表示狗，1表示猫
            file = os.listdir(full_filename)   #["0", "1"]
            #print(file)
            for name in file:
                #获得train文件夹下的子文件夹“0”和“1”的绝对路径
                temp = os.path.join(full_filename, name)
                for img in os.listdir(temp):    #分别遍历两个文件夹["0", "1"]下的所有图像
                    #将图像的信息写入到train.txt文件
                    file_val.write(root_data_path + "\\"+ name + "\\" + img + " " + name + "\n")
        else:
            print("存在train val以外的文件")
else:
    print("该文件夹下不存在子文件夹")

file_train.close()
file_val.close()