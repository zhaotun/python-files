import cv2
from PIL import Image
import numpy as np
import os

root_path = r"F:\binlang\srcImgs\small\test"
final_path = r"F:\binlang\srcImgs\small\test_res"

yuantu = "L1.jpeg"
masktu = "L1_jpeg_recover.png"

#使用opencv叠加图片
#img1 = cv2.imread(yuantu)
#img2 = cv2.imread(masktu)

alpha = 0.5
meta = 1 - alpha
gamma = 0

i=0

for root, dirs, files in os.walk(root_path):  # walk 遍历当前source_path目录和所有子目录的文件和目录
   for file in files:                          # files 是所有的文件名列表，
       if file.endswith('jpeg'):
          filename = file.replace('.jpeg','')
          filepath = os.path.join(root, file)
          #print(filename)
          #print(filepath)
          srcimg = cv2.imread(filepath)
 
          maskfilename = filename + '_jpeg_recover.png'
          maskfilepath = os.path.join(root, maskfilename)
          #print(maskfilename)
          #print(maskfilepath)
          maskimg = cv2.imread(maskfilepath)

          maskedimage = cv2.add(srcimg, maskimg)
          #cv2.imshow('maskedimage', maskedimage)
          #cv2.waitKey(0)
          #cv2.destroyAllWindows()

          maskedfilename = filename + '_Masked.png'
          maskedfilepath = os.path.join(final_path, maskedfilename)
          cv2.imwrite(maskedfilepath,maskedimage)

