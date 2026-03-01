import cv2 as cv
import numpy as np
import math

CHECKED_POINTS = 100
CIRCLE_POINTS = 40

class ArcProcessor:
    def check_circle(self, edges, x, y, radius):
        angles = np.linspace(0, 2 * np.pi, CHECKED_POINTS)
    
        x_coords = (x + radius * np.cos(angles)).astype(int)
        y_coords = (y + radius * np.sin(angles)).astype(int)

        valid_mask = (x_coords >= 0) & (x_coords < edges.shape[1]) & \
                 (y_coords >= 0) & (y_coords < edges.shape[0])

        mask_angles = np.zeros(CHECKED_POINTS, dtype=int)

        mask_angles[valid_mask] = (edges[y_coords[valid_mask], x_coords[valid_mask]] > 0)
        
        total_points = np.sum(mask_angles)
    
        if total_points > CIRCLE_POINTS:
            return (x, y, radius, -1, -1)
        extended = np.concatenate([mask_angles, mask_angles])
        max_len = 0
        best_start = -1
        current_len = 0
        current_start = 0

        for i in range(len(extended)):
            if extended[i] == 1:
                if current_len == 0:
                    current_start = i
                current_len += 1
                if current_len > max_len:
                    max_len = current_len
                    best_start = current_start
            else:
                current_len = 0

        if max_len == 0:
            return (x, y, radius, -1, -1)

        start = best_start % CHECKED_POINTS
        end = (best_start + max_len) % CHECKED_POINTS

        start_angle = start * (2 * np.pi) / CHECKED_POINTS
        end_angle = end * (2 * np.pi) / CHECKED_POINTS

        return (x, y, radius, start_angle, end_angle)
        
    def process(self, edges, circles):
        kernel = np.ones((21,21), np.uint8)
        edges = cv.dilate(edges, kernel, iterations=1)
        output_circles = []
        output_arcs = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                arc = self.check_circle(edges, i[0], i[1], i[2])
                if arc[3] == -1:
                    output_circles.append((arc[0], arc[1], arc[2]))
                else:
                    output_arcs.append(arc)
        return output_circles, output_arcs
    
    def draw(self, frame, arcs):
        for arc in arcs:
            (x, y, radius, start_angle, end_angle) = arc
            cv.ellipse(frame, (x,y), (radius, radius), 0, start_angle, end_angle, (0, 255, 255), 3)
        return frame

