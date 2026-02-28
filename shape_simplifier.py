import numpy as np

class ShapeSimplifier:
    length_threshold = 0
    dist_threshold = 0

    def __init__(self, length_threshold, dist_threshold):
        self.length_threshold = length_threshold
        self.dist_threshold = dist_threshold

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