import cv2 as cv
import numpy as np

class TagDetector:
    detector_params = cv.aruco.DetectorParameters()
    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_APRILTAG_16h5)

    def __init__(self):
        self.detector_params.cornerRefinementMethod = cv.aruco.CORNER_REFINE_APRILTAG
        self.detector_params.aprilTagQuadDecimate = -1 # TODO: set
        self.detector_params.aprilTagQuadSigma = -1 # TODO: set
        


        pass

    def detect(self, frame):
        pass