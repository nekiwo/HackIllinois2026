import cv2 as cv
import numpy as np

class LineDetector:
    def detect(self, img, edges):
        lines = cv.HoughLinesP(edges,1,np.pi/180,100,minLineLength=100,maxLineGap=10)
        if lines is None:
            return
        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv.line(img,(x1,y1),(x2,y2),(0,255,0),2)