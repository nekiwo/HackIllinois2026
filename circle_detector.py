import cv2 as cv
import numpy as np

MIN_RADIUS=1
MAX_RADIUS=80

class CircleDetector:
    def detect(self, gray, num_iterations=4):
        all_circles = np.empty(1)
        group_size = (MAX_RADIUS- MIN_RADIUS) / num_iterations
        for i in range(num_iterations):
            rows = gray.shape[0]
            circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 16,
                                param1=100, param2=15, minRadius=int(group_size*i + MIN_RADIUS), 
                                maxRadius=int(group_size*(i+1) + MIN_RADIUS))
            if circles is None:
                continue
            if all_circles.ndim < 2:
                all_circles = circles
            else:
                np.concatenate((all_circles, circles), axis=1)
        if all_circles.ndim <2:
            return None
        return all_circles
    
    def draw(self, frame, circles):
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                cv.circle(frame, center, 1, (0, 100, 100), 8)
                radius = i[2]
                cv.circle(frame, center, radius, (255, 0, 255), 8)
        return frame