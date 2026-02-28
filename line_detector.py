import cv2 as cv
import numpy as np

class LineDetector:
    fld = None
    def __init__(self, canny_th1, canny_th2, canny_aperture):
        self.fld = cv.ximgproc.createFastLineDetector(canny_th1=canny_th1, canny_th2=canny_th2, 
                                                 canny_aperture_size=canny_aperture, do_merge = True)
        
    def draw_lines(self, frame, lines):
        return self.fld.drawSegments(frame, lines, linethickness=8)
    
    def detect(self, frame, gray):
        lines = self.fld.detect(gray)
        if lines is None:
            return []
        return lines