# -*- coding:utf-8 -*-

'''
获取图片HSV的最大、最小、平均值
'''
  
import cv2
import numpy as np 

def image_traverse_get_hsv_values( hsv_img ):
    
    pixels_list = []

    height = hsv_img.shape[0]        #将tuple中的元素取出，赋值给height，width，channels
    width  = hsv_img.shape[1]
    channels = hsv_img.shape[2]
    
    """遍历图像每个像素的每个通道"""
    for row in range(height):    #遍历每一行
        for col in range(width): #遍历每一列                    
            h = hsv_img[row][col][0]
            s = hsv_img[row][col][1]
            v = hsv_img[row][col][2]
            #print('h,s,v = ',h,s,v)
            
            pixels_list.append((h,s,v)) # 将每个点的HSV值保存到List

    npp = np.array(pixels_list) 
    
    hsv_min  = npp.min(axis=0)
    hsv_max  = npp.max(axis=0)
    hsv_mean = npp.mean(axis=0)    
   
    return hsv_min,hsv_max,hsv_mean
   
 
if __name__ == '__main__':
        
    img_src = cv2.imread('ROI.jpg')
    
    img_hsv = cv2.cvtColor( img_src,cv2.COLOR_BGR2HSV)
    
    hsv_min,hsv_max,hsv_mean = image_traverse_get_hsv_values( img_hsv )

    print('hsv_min  = ', hsv_min )
    print('hsv_max  = ', hsv_max )
    print('hsv_mean = ', hsv_mean)