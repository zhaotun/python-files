import cv2
import os
import random
import shutil

    
if __name__ == '__main__':
    
    txt_file_path = r'D:\tiff\data\croped_images\road_seg_dataset\test_list.txt'
    
    src_imgs_path = r'D:\tiff\data\croped_images\road_seg_dataset'
    
    with open(txt_file_path, "r", encoding="utf-8") as txt_file:
        for line in txt_file:
            line_info = line.split(' ')
            print('line_info = ',line_info)
            
            src_img = os.path.join(src_imgs_path, line_info[0])
            dst_img = os.path.join(src_imgs_path, line_info[0].replace('JPEGImages/','test_imgs/'))
            
            shutil.copy(src_img, dst_img)