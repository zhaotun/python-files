import cv2
import numpy as np

'''
通过设定HSV范围提取相应颜色的mask，并保存
'''


if __name__ == '__main__':

    img_original = cv2.imread('2.png')
    
    #颜色空间的转换
    img_hsv = cv2.cvtColor(img_original,cv2.COLOR_BGR2HSV)
    
    # HSV范围设置
    #lower_hsv = np.array([0, 0, 60])
    #upper_hsv = np.array([255, 255, 255])
 
    lower_hsv = np.array([80, 28, 28])
    upper_hsv = np.array([80, 55, 53])
    
    mask = cv2.inRange(img_hsv, lowerb=lower_hsv, upperb=upper_hsv) # 选定颜色置白
    #ROI掩模区域反向掩模
    #mask = cv2.bitwise_not(mask) # 选定颜色置黑
    cv2.imshow("mask0", mask)
    cv2.imwrite("mask.jpg", mask)

    cv2.waitKey(0)