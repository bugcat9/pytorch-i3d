import cv2 as cv
import numpy as np
import os
path = 'origi_videos'
videos = os.listdir(path)
TVL1 = cv.optflow.DualTVL1OpticalFlow_create()
for video in videos:
    video_path = path + '/' + video
    cap = cv.VideoCapture(video_path)
    ret, frame1 = cap.read()
    prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255
    count = 0
    while(True):
        ret, frame2 = cap.read()
        if not ret:
            break
        next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
 
        # 返回一个两通道的光流向量，实际上是每个点的像素位移值
        flow = TVL1.calc(prvs,next, None)
 
        # 笛卡尔坐标转换为极坐标，获得极轴和极角
        mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
 
        hsv[...,0] = ang*180/np.pi/2  #角度
        hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
        bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
        path_ = path + '/' + os.path.basename(video).split('.')[0]
        if not os.path.exists(path_):
            os.makedirs(path_)
        cv.imwrite(path_ + "/frame{0:06d}.jpg".format(count), bgr)
        count += 1
        prvs = next
    cap.release()
    cv.destroyAllWindows()
