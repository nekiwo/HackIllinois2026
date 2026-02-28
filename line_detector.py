import cv2 as cv
import numpy as np

class LineDetector:
    fld = None
    def __init__(self, canny_th1, canny_th2, canny_aperture):
        self.fld = cv.ximgproc.createFastLineDetector(canny_th1=canny_th1, canny_th2=canny_th2, 
                                                 canny_aperture_size=canny_aperture, do_merge = True)
    
    def detect(self, frame, gray):
        lines = self.fld.detect(gray)
        if lines is None:
            return
        draw_frame = self.fld.drawSegments(frame, lines, linethickness=8)
        return draw_frame
