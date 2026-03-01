import numpy as np

class ShapeSimplifier:
    length_threshold = 0
    dist_threshold = 0
    circle_clean_threshold = 0

    def __init__(self, length_threshold, dist_threshold, circle_clean_threshold):
        self.length_threshold = length_threshold
        self.dist_threshold = dist_threshold
        self.circle_clean_threshold = circle_clean_threshold

    def get_p(self, line, p_i):
        return np.array([line[0][2 * p_i + 0], line[0][2 * p_i + 1]])
    
    def set_p(self, line, point, p_i):
        line[0][2 * p_i + 0] = point[0]
        line[0][2 * p_i + 1] = point[1]
        return line
    
    def get_dist(self, line):
        return np.linalg.norm(self.get_p(line, 1) - self.get_p(line, 0))
    
    def get_dist_2p(self, p1, p2):
        return np.linalg.norm(p2 - p1)

    def simplify(self, lines):
        while True:
            has_changes = False
            for line_i, line in enumerate(lines):
                # if self.get_dist(line) <= self.length_threshold:
                #     lines = np.delete(lines, line_i)
                #     has_changes = True
                #     continue

                for point_i in range(2):
                    point = self.get_p(line, point_i)
                    for line2_i, line2 in enumerate(lines):
                        if line_i == line2_i:
                            continue

                        for point2_i in range(2):
                            point2 = self.get_p(line2, point2_i)
                            new_point = None
                            dist = self.get_dist_2p(point, point2)
                            if dist > 0.5 and dist <= self.dist_threshold:
                                new_point = ((point + point2) / 2.0).astype(np.int32)
                                lines[line_i] = self.set_p(line, new_point, point_i)
                                lines[line2_i] = self.set_p(line2, new_point, point2_i)
                                has_changes = True

            if not has_changes:
                break

        return lines
    
    def remove_apriltag(self, lines, circles, x_bound, y_bound):
        line_i = 0
        while line_i < len(lines):
            point = self.get_p(lines[line_i], 0)
            if point[0] >= x_bound and point[1] <= y_bound:
                lines = np.delete(lines, line_i, axis=0)
                line_i -= 1
                continue
            line_i += 1

        circle_i = 0
        circles = np.uint16(np.around(circles))[0]
        while circle_i < len(circles):
            origin = [circles[circle_i][0], circles[circle_i][1]]
            if origin[0] >= x_bound and origin[1] <= y_bound:
                circles = np.delete(circles, circle_i, axis=0)
                circle_i -= 1
                continue
            circle_i += 1

        circles = [circles]
        return lines, circles
    
    def clean_circles(self, lines, circles):
        circles = np.uint16(np.around(circles))[0]
        line_i = 0
        while line_i < len(lines):
            line = lines[line_i]
            is_deleted = False
            for circle in circles:
                center = [circle[0], circle[1]]
                radius = circle[2]
                point1 = self.get_p(line, 0)
                point2 = self.get_p(line, 1)
                if np.linalg.norm(point1 - center) <= radius + self.circle_clean_threshold or np.linalg.norm(point2 - center) <= radius + self.circle_clean_threshold:
                    lines = np.delete(lines, line_i, axis=0)
                    line_i -= 1
                    is_deleted = True
                    break
            if is_deleted:
                continue
            line_i += 1
        return lines