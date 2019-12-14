import os, random, shutil
import cv2


#源图片文件夹路径
a_path  = r'F:\ImageSegmentation\PaddleSeg\dataset\binglang_train_1119\SegmentationClassPNG'
a_test  = r'F:\ImageSegmentation\PaddleSeg\dataset\binglang_train_1119\annotations\test'
a_val   = r'F:\ImageSegmentation\PaddleSeg\dataset\binglang_train_1119\annotations\val'
a_train = r'F:\ImageSegmentation\PaddleSeg\dataset\binglang_train_1119\annotations\train'

i_path  = r'F:\ImageSegmentation\PaddleSeg\dataset\binglang_train_1119\JPEGImages'
i_test  = r'F:\ImageSegmentation\PaddleSeg\dataset\binglang_train_1119\images\test'
i_val   = r'F:\ImageSegmentation\PaddleSeg\dataset\binglang_train_1119\images\val'
i_train = r'F:\ImageSegmentation\PaddleSeg\dataset\binglang_train_1119\images\train'

i=0
for root, dirs, files in os.walk(a_test):  # walk 遍历当前source_path目录和所有子目录的文件和目录
        for file in files:                          # files 是所有的文件名列表，
            pngfile = file.replace("png","jpeg")
            shutil.move(os.path.join(i_path, pngfile), os.path.join(i_test, pngfile))
            i=i+1
print("%d i_test files been moved !",i)

i=0
for root, dirs, files in os.walk(a_val):  # walk 遍历当前source_path目录和所有子目录的文件和目录
        for file in files:                          # files 是所有的文件名列表，
            pngfile = file.replace("png","jpeg")
            shutil.move(os.path.join(i_path, pngfile), os.path.join(i_val, pngfile))
            i=i+1
print("%d i_val files been moved !",i)

i=0
for root, dirs, files in os.walk(a_train):  # walk 遍历当前source_path目录和所有子目录的文件和目录
        for file in files:                          # files 是所有的文件名列表，
            pngfile = file.replace("png","jpeg")
            shutil.move(os.path.join(i_path, pngfile), os.path.join(i_train, pngfile))
            i=i+1
print("%d i_train files been moved !",i)

