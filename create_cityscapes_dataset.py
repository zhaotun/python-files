#!/usr/bin/python3
# -*- coding: utf-8 -*

# @Description   : 从源目录中随机选择指定数量的文件，并将它们移动到目标目录中，若目标目录中已存在相应文件，则不移动
# @Author        : wt
# @Date          : 2021-6-25 17:58
# @Run           : python random_move_files_from_src2dst.py

import os
import shutil

def random_move_files_from_srcpath_to_dstpath(SRC_PATH, DST_PATH, PICKED_NUM):
    import random,shutil,os
    
    src_files = os.listdir(SRC_PATH)
    dst_files = os.listdir(DST_PATH)
    
    # 从源目录中随机选择指定数量的文件
    selected_files = random.sample(src_files, PICKED_NUM)
    #print('selected_files = ',selected_files)
    
    for i in range( len(selected_files) ):
        if selected_files[i] in dst_files:
            print(DST_PATH, ' 目录中已存在 ',selected_files[i],' 文件。')
            continue
        
        SRC_FILE = os.path.join(SRC_PATH, selected_files[i])
        shutil.move(SRC_FILE, DST_PATH)
        print('文件 ',selected_files[i],' 已被移动到目录',DST_PATH)

  
if __name__ == "__main__":

    train_rate = 0.8
    val_rate   = 0.1
    test_rate  = 0.1
    
    SRC_IMAGES_PATH = r'E:\datasets\road-seg\UAS_Dataset\uas_road\images'
    SRC_LABELS_PATH = r'E:\datasets\road-seg\UAS_Dataset\uas_road\labels'
    
    DST_CITYSCAPES_PATH = r'E:\datasets\road-seg\UAS_Dataset\uas_road_cityscapes'
    
    gtFine_train = DST_CITYSCAPES_PATH + '/gtFine/train'
    gtFine_val   = DST_CITYSCAPES_PATH + '/gtFine/val'
    gtFine_test  = DST_CITYSCAPES_PATH + '/gtFine/test'
    
    leftImg8bit_train = DST_CITYSCAPES_PATH + '/leftImg8bit/train'
    leftImg8bit_val   = DST_CITYSCAPES_PATH + '/leftImg8bit/val'
    leftImg8bit_test  = DST_CITYSCAPES_PATH + '/leftImg8bit/test'
    
    if not os.path.exists(gtFine_train):
        os.makedirs(gtFine_train)
    if not os.path.exists(gtFine_val):
        os.makedirs(gtFine_val)
    if not os.path.exists(gtFine_test):
        os.makedirs(gtFine_test)
        
    if not os.path.exists(leftImg8bit_train):
        os.makedirs(leftImg8bit_train)
    if not os.path.exists(leftImg8bit_val):
        os.makedirs(leftImg8bit_val)
    if not os.path.exists(leftImg8bit_test):
        os.makedirs(leftImg8bit_test)
    
    src_files = os.listdir(SRC_IMAGES_PATH)
    total_cnt = len(src_files)
    
    train_cnt = (int)(total_cnt * train_rate)
    val_cnt   = (int)(total_cnt * val_rate)
    test_cnt  = total_cnt - train_cnt - val_cnt
    
    train_files = src_files[  : train_cnt]
    val_files   = src_files[ train_cnt : train_cnt + val_cnt]
    test_files  = src_files[ train_cnt + val_cnt : ]
    
    print(len(train_files))
    print(len(val_files))
    print(len(test_files))
    
    for file in train_files:
        src_file_path = os.path.join(SRC_IMAGES_PATH, file)
        dst_file_path = os.path.join(leftImg8bit_train, file)
        shutil.copy(src_file_path, dst_file_path)
        
        src_file_path = os.path.join(SRC_LABELS_PATH, file.replace('.jpg','.png'))
        dst_file_path = os.path.join(gtFine_train, file.replace('.jpg','.png'))
        shutil.copy(src_file_path, dst_file_path)
        
    for file in val_files:
        src_file_path = os.path.join(SRC_IMAGES_PATH, file)
        dst_file_path = os.path.join(leftImg8bit_val, file)
        shutil.copy(src_file_path, dst_file_path)
        
        src_file_path = os.path.join(SRC_LABELS_PATH, file.replace('.jpg','.png'))
        dst_file_path = os.path.join(gtFine_val, file.replace('.jpg','.png'))
        shutil.copy(src_file_path, dst_file_path)
    
    for file in test_files:
        src_file_path = os.path.join(SRC_IMAGES_PATH, file)
        dst_file_path = os.path.join(leftImg8bit_test, file)
        shutil.copy(src_file_path, dst_file_path)
        
        src_file_path = os.path.join(SRC_LABELS_PATH, file.replace('.jpg','.png'))
        dst_file_path = os.path.join(gtFine_test, file.replace('.jpg','.png'))
        shutil.copy(src_file_path, dst_file_path)
        
    #random_move_files_from_srcpath_to_dstpath(SRC_PATH, DST_PATH, PICKED_NUM)
    

    
    
  
