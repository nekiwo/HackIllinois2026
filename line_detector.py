import cv2 as cv
import numpy as np

class LineDetector:
    fld = None
    def __init__(self, canny_th1, canny_th2, canny_aperture):
        self.fld = cv.ximgproc.createFastLineDetector(canny_th1=canny_th1, canny_th2=canny_th2, 
                                                 canny_aperture_size=canny_aperture, do_merge = True)
    
    def detect(self, frame, gray, subsample_percent):
        resize = cv.resize(gray, None, fx=subsample_percent, fy=subsample_percent, interpolation=cv.INTER_AREA)
        #blurred = cv.GaussianBlur(resize, (3, 3), 0)
        lines = self.fld.detect(resize)
        if lines is None:
            return
        scale = (1/subsample_percent)
        lines = lines * scale
        draw_frame = self.fld.drawSegments(frame, lines, linethickness=8)
        return draw_frame, lines
