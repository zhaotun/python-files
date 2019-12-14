#!/usr/bin/python
import cv2
import time 
import sys

cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(2)

cap0.set(3,640)
cap0.set(4,480)
cap0.set(1, 10.0)

cap1.set(3,640)
cap1.set(4,480)
cap1.set(1, 10.0)

cap2.set(3,640)
cap2.set(4,480)
cap2.set(1, 10.0)

time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
print(time)

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out0 = cv2.VideoWriter('./video/camera-0-'+time+'.avi',fourcc, 10, (640, 480))
out1 = cv2.VideoWriter('./video/camera-1-'+time+'.avi',fourcc, 10, (640, 480))
out2 = cv2.VideoWriter('./video/camera-2-'+time+'.avi',fourcc, 10, (640, 480))

while(1):

    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    #cv2.imshow("cap-0", frame0)
    #cv2.imshow("cap-1", frame1)
    #cv2.imshow("cap-2", frame2)

    #if ret0 == True:
    #    frame0 = cv2.flip(frame0, 1)
    #    a = out0.write(frame0)
    #    cv2.imshow("cap-0", frame0)
    #    if cv2.waitKey(1) & 0xFF == ord('q'):
    #        break
    #else:
    #    break

    if ret1 == True:
        frame1 = cv2.flip(frame1, 1)
        b = out2.write(frame1)
        cv2.imshow("cap-1", frame1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

    if ret2 == True:
        frame2 = cv2.flip(frame2, 1)
        c = out2.write(frame2)
        cv2.imshow("cap-2", frame2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):

        cv2.imwrite("camera.jpg", frame0)
        break

cap0.release()
cap1.release()
cap2.release()
cv2.destroyAllWindows()
