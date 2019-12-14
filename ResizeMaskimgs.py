import os, random, shutil
import cv2


#源图片文件夹路径
source_path       = r'F:\ImageSegmentation\PaddleSeg\dataset\binglang_train_1119\images_1483'
mask_path         = r'F:\binlang\srcImgs\small\test_png'
resized_mask_path = r'F:\ImageSegmentation\PaddleSeg\dataset\binglang_train_1119\annotations_1483\trimaps'

i=0
if os.path.exists(source_path):
    for root, dirs, files in os.walk(source_path):  # walk 遍历当前source_path目录和所有子目录的文件和目录
        for file in files:                          # files 是所有的文件名列表，
            #if file.endswith('_jpeg.png') or file.endswith('_jpeg_scoremap.png'):
            maskname = file.replace('.','_')
            maskname = maskname + ".png"
            maskimg = os.path.join(mask_path, maskname)
            print(maskimg)
            mask_img = cv2.imread(maskimg)
            #mask_resize_img = mask_img
            #cv2.imshow("before resize",mask_img)

            srcimgname = os.path.join(root, file)
            src_img    = cv2.imread(srcimgname)
            #cv2.imshow("src_img",src_img)
            #print(srcimgname)

            x, y = src_img.shape[0:2]
            mask_resize_img = cv2.resize(mask_img,(y,x), 0, 0, cv2.INTER_NEAREST)
            #cv2.imshow("after resize",mask_resize_img)
            labeled_imgname = os.path.join(resized_mask_path, file.replace('jpeg','png') )
            cv2.imwrite( labeled_imgname,mask_resize_img)
            i= i+1

print("%d mask imgs been resized and stored ...",i)













	
