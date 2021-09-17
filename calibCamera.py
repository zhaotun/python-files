# -*- coding: utf-8 -*-
"""
Homework: Calibrate the Camera with ZhangZhengyou Method.
Picture File Folder: ".\pic\IR_camera_calib_img", With Distort. 

By YouZhiyuan 2019.11.18
"""

import os
import numpy as np
import cv2
import glob

# https://blog.csdn.net/u010128736/article/details/52875137

def calib(inter_corner_shape, size_per_grid, img_dir, img_type):
    # criteria: only for subpix calibration, which is not used here.
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    w,h = inter_corner_shape
    # cp_int: corner point in int form, save the coordinate of corner points in world sapce in 'int' form
    # like (0,0,0), (1,0,0), (2,0,0) ....,(10,7,0).
    cp_int = np.zeros((w*h,3), np.float32)
    cp_int[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
    # cp_world: corner point in world space, save the coordinate of corner points in world space.
    cp_world = cp_int*size_per_grid # 计算世界坐标，z为0
    print('\n\n------------------ 真实世界坐标 --------------------\n\n')
    print('cp_world.shape = ',cp_world.shape)
    print('cp_world[:3]   = \n',cp_world[:3] )
    
    obj_points = [] # the points in world space
    img_points = [] # the points in image space (relevant to obj_points)
    images = glob.glob(img_dir + os.sep + '**.' + img_type)
    for fname in images:
        img = cv2.imread(fname)
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # find the corners, cp_img: corner points in pixel space.
        # 检测棋盘格角点，输出像素坐标
        ret, cp_img = cv2.findChessboardCorners(gray_img, (w,h), None)
        # if ret is True, save.
        if ret == True:
            # 获得更为准确的角点像素坐标
            # cv2.cornerSubPix(gray_img,cp_img,(11,11),(-1,-1),criteria)
            obj_points.append(cp_world) # 保存世界坐标
            img_points.append(cp_img)   # 保存像素坐标
            # view the corners
            cv2.drawChessboardCorners(img, (w,h), cp_img, ret)
            #cv2.imshow('FoundCorners',img)
            #cv2.imwrite( fname + '_fc.jpg',img)
            #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    # calibrate the camera
    # 开始标定
    ret, mat_inter, coff_dis, v_rot, v_trans = cv2.calibrateCamera(obj_points, img_points, gray_img.shape[::-1], None, None)
    print('\n\n------------------ 经过标定得出的参数 --------------------\n\n')
    # 重投影误差
    print ("重投影误差 ret: \n",ret)
    # 相机内参矩阵
    print ("\n相机内参矩阵 internal matrix :\n" ,mat_inter)
    # 相机畸变参数
    # in the form of (k_1,k_2,p_1,p_2,k_3)
    print ("\n相机畸变系数 distortion cofficients, coff_dis :\n",coff_dis)  
    # 世界坐标到相机坐标的旋转参数，需要进行罗德里格斯转换 https://blog.csdn.net/qq_40475529/article/details/89409303
    #print (("旋转矩阵 rotation vectors:\n"),v_rot)
    #print ("\n旋转向量 len(v_rot):",len(v_rot))
    print ("\n旋转向量 rotation v_rot[0]: \n",v_rot[0])
    
    # 旋转向量转旋转矩阵
    om = np.array([-0.34620865, 0.1342627, -0.06781741 ])
    rot_matrix = cv2.Rodrigues(om)[0]
    #print ("旋转矩阵 len(rot_matrix):",len(rot_matrix))
    print ("\n Rodrigues转换的旋转矩阵 rotation rot_matrix[0]: \n",rot_matrix)
    
    # 平移参数
    print ("\n 平移向量 v_trans: \n",len(v_trans))
    print ("\n 平移向量 translation v_trans[0]: \n",v_trans[0])
    # calculate the error of reproject
    total_error = 0
    for i in range(len(obj_points)):
        # 计算反投影误差，通过给定的内参数和外参数计算三维点投影到二维图像平面上的坐标
        img_points_repro, _ = cv2.projectPoints(obj_points[i], v_rot[i], v_trans[i], mat_inter, coff_dis)
        #print('------ len(obj_points) : ',len(obj_points)) # 世界坐标
        #print('------ len(img_points_repro) : ',len(img_points_repro))
        print('\n\n------------------ 根据标定得出的参数进行反投影：',i,' --------------------\n\n')
        print('世界坐标：\n',obj_points[i][:3])
        print('\n内参矩阵：\n',mat_inter)
        print('\n畸变参数：\n',coff_dis)
        print('\n旋转向量：\n',v_rot[i])
        
        a = v_rot[i][0][0]
        b = v_rot[i][1][0]
        c = v_rot[i][2][0]
        
        rot_matrix = cv2.Rodrigues( np.array( [a,b,c] ) )[0]
        print('\n经Rodrigues转换的旋转矩阵：\n',rot_matrix)
        
        print('\n平移向量：\n',v_trans[i])
        print('\n通过图像检测的像素坐标：\n',img_points[i][:3])
        print('\n计算转换后的像素坐标：\n',img_points_repro[:3]) # 转换后的像素坐标
        error = cv2.norm(img_points[i], img_points_repro, cv2.NORM_L2)/len(img_points_repro)
        print('\n转换误差：\n',error)
        total_error += error
    print(("\n平均反投影误差 Average Error of Reproject: \n"), total_error/len(obj_points))
    
    return mat_inter, coff_dis
  
# 根据内参和畸变系数消除畸变  
def dedistortion(inter_corner_shape, img_dir,img_type, save_dir, mat_inter, coff_dis):
    w,h = inter_corner_shape
    images = glob.glob(img_dir + os.sep + '**.' + img_type)
    for fname in images:
        img_name = fname.split(os.sep)[-1]
        img = cv2.imread(fname)
        # 优化内参数和畸变系数
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mat_inter,coff_dis,(w,h),0,(w,h)) # 自由比例参数
        # 去畸变
        dst = cv2.undistort(img, mat_inter, coff_dis, None, newcameramtx)
        # clip the image
        # x,y,w,h = roi
        # dst = dst[y:y+h, x:x+w]
        cv2.imwrite(save_dir + os.sep + img_name, dst)
    print('Dedistorted images have been saved to: %s successfully.' %save_dir)
    
if __name__ == '__main__':
    
    inter_corner_shape = (8,6) # 棋盘格角点矩阵
    size_per_grid = 24.5       # 棋盘格每格实际尺寸，单位为mm
    img_dir = "./pic/iphone_pic"
    img_type = "jpg"
    # calibrate the camera
    mat_inter, coff_dis = calib(inter_corner_shape, size_per_grid, img_dir,img_type)
    # dedistort and save the dedistortion result. 
    save_dir = "./pic/iphone_pic_saved"
    if(not os.path.exists(save_dir)):
        os.makedirs(save_dir)
    dedistortion(inter_corner_shape, img_dir, img_type, save_dir, mat_inter, coff_dis)
    
    
    