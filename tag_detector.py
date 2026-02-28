import cv2 as cv
import numpy as np

class TagDetector:
    detector_params = cv.aruco.DetectorParameters()
    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_APRILTAG_36h11)
    aruco_detector = None
    obj_points = np.float32([
            [-0.5, 0.5, 0],
            [0.5, 0.5, 0],
            [0.5, -0.5, 0],
            [-0.5, -0.5, 0]
        ])

    def __init__(self):
        self.detector_params.cornerRefinementMethod = cv.aruco.CORNER_REFINE_APRILTAG
        self.detector_params.aprilTagQuadDecimate = -1 # TODO: set
        self.detector_params.aprilTagQuadSigma = -1 # TODO: set
        self.aruco_detector = cv.aruco.ArucoDetector(self.dictionary, self.detector_params)

    def detect(self, frame):
        corners, ids, rejected = self.aruco_detector.detectMarkers(frame)
        return ids, corners
    
    def estimate_pose(self, corners, camera_mat, dist_coeffs):
        ret, rvecs, tvecs = cv.solvePnP(self.obj_points, corners, camera_mat, dist_coeffs, flags=cv.SOLVEPNP_IPPE_SQUARE)
        return 0