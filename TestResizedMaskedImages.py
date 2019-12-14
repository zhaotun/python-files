import os, random, shutil
import cv2


source_path = r'F:\ImageSegmentation\PaddleSeg\dataset\binglang_train_1119\images_1483'
resized_mask_path = r'F:\ImageSegmentation\PaddleSeg\dataset\binglang_train_1119\annotations_1483\trimaps'

res_path = r'F:\ImageSegmentation\PaddleSeg\dataset\binglang_train_1119\testmask'

if os.path.exists(source_path):
    for root, dirs, files in os.walk(source_path):  # walk 遍历当前source_path目录和所有子目录的文件和目录
        for file in files:                          # files 是所有的文件名列表，
            #if file.endswith('_jpeg.png') or file.endswith('_jpeg_scoremap.png'):
            maskname = file.replace('jpeg','png')
            maskimg = os.path.join(resized_mask_path, maskname)

            mask_img = cv2.imread(maskimg)
            #print(mask_img.shape) 
            height   = mask_img.shape[0]
            weight   = mask_img.shape[1]
            channels = mask_img.shape[2]
            #print("weight : %s, height : %s, channel : %s" %(weight, height, channels))
            for row in range(height):            #遍历高
               for col in range(weight):         #遍历宽
                     pv = mask_img[row, col, 0] 
                     if pv == 1: 
                        mask_img[row, col, 0] = 255
                        mask_img[row, col, 1] = 255
                        mask_img[row, col, 2] = 255

            srcimgname = os.path.join(root, file)
            src_img    = cv2.imread(srcimgname)
            #cv2.imshow("src_img",src_img)
            #print(srcimgname)

            maskedimage = cv2.add(src_img, mask_img)
            #cv2.imshow("masked img",maskedimage)

            finalpath = os.path.join(res_path, maskname)
            cv2.imwrite(finalpath,maskedimage)

            cv2.waitKey(0)











	
