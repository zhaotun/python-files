# -*- coding:utf-8 -*-

'''
通过鼠标左键截取图片中的ROI区域，并将其保存为图片
'''

import cv2
 
#global img, cut_img
#global point1, point2

def on_mouse(event, x, y, flags, param):

    global img, point1, point2, cut_img
    img2 = img.copy()
    
    #左键点击
    if event == cv2.EVENT_LBUTTONDOWN:         
        point1 = (x,y)
        cv2.circle(img2, point1, 10, (0,255,0), 1)
        cv2.imshow('image', img2)
    
    #按住左键拖曳    
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):               
        cv2.rectangle(img2, point1, (x,y), (255,0,0), 1)
        cv2.imshow('image', img2)
    
    #左键释放
    elif event == cv2.EVENT_LBUTTONUP:         
        point2 = (x,y)
        cv2.rectangle(img2, point1, point2, (0,0,255), 1) 
        cv2.imshow('image', img2)
        min_x = min(point1[0],point2[0])     
        min_y = min(point1[1],point2[1])
        width = abs(point1[0] - point2[0])
        height = abs(point1[1] -point2[1])
        cut_img = img[min_y:min_y+height, min_x:min_x+width]
        cv2.imwrite('ROI.jpg', cut_img)

 
def main():
    global img,cut_img
    img = cv2.imread('2.png')
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', on_mouse)
    cv2.imshow('image', img)
    #cv2.imshow('cut_img', cut_img)
    cv2.waitKey(0)
 
if __name__ == '__main__':
    main()