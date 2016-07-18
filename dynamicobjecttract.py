import cv2
import numpy as np

#cap = cv2.VideoCapture("E://PYTHON//Opencv//createVideo//VideoRecoder.avi")

cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret,frame = cap.read()
    frame = cv2.resize(frame,(800,600))
    #print(ret)        
    fgmask = fgbg.apply(frame)
    fgmask = cv2.equalizeHist(fgmask)
    blurmask = cv2.GaussianBlur(fgmask,(21,21),0)
    thresh = cv2.threshold(blurmask,100,255,0)[1]
    #thresh = cv2.dilate(thresh,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,4)),iterations = 2)
    image,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    img = cv2.drawContours(frame.copy(),contours,0,(0,255,0),2)
    for con in contours:
        x,y,w,h = cv2.boundingRect(con)
        if w*h > 10000:#绘制的矩形最小面积，以像素为单位
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
        else:
            pass
    cv2.putText(frame,"Press Q to quit",(10,30),cv2.FORMATTER_FMT_PYTHON,1,(0,0,255),2)
    cv2.imshow("FG",fgmask)
    cv2.imshow("Track",frame)
    if cv2.waitKey(1)&0xff == ord("q"):
        cap.release()
        break
cv2.destroyAllWindows()
    
