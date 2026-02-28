import cv2 as cv
import numpy as np

class LineDetector:
    fld = cv.ximgproc.createFastLineDetector(canny_th1 = 50.0, canny_th2 = 150.0, canny_aperture_size = 3, do_merge = True)
    def detect(self, frame, gray):
        lines = self.fld.detect(gray)
        if lines is None:
            return
        draw_frame = self.fld.drawSegments(frame, lines, linethickness=8)
        return draw_frame, lines
