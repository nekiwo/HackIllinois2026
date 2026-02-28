import cv2 as cv
import numpy as np

class CircleDetector:
    def detect(self, frame, gray):
        rows = gray.shape[0]
        circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
                               param1=75, param2=25,
                               minRadius=1, maxRadius=100)
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                cv.circle(frame, center, 1, (0, 100, 100), 3)
                radius = i[2]
                cv.circle(frame, center, radius, (255, 0, 255), 3)
        return frame, circles