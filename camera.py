#!/usr/bin/python
import cv2

cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(2)

while(1):
 # 获得图片
    ret, frame0 = cap0.read()
    ret, frame1 = cap1.read()
    ret, frame2 = cap2.read()
    # 展示图片
    cv2.imshow("cap-0", frame0)
    cv2.imshow("cap-1", frame1)
    cv2.imshow("cap-2", frame2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        # 存储图片
        cv2.imwrite("camera.jpg", frame0)
        break

cap0.release()
cap1.release()
cap2.release()
cv2.destroyAllWindows()
