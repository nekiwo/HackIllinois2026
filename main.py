import cv2 as cv
import numpy as np

from tag_detector import TagDetector
from line_detector import LineDetector

# Config
DEBUG = True
# PS Eye
camera_mat = np.matrix([
    [1, 0, 0],
    [0, 1, 1],
    [0, 0, 1]
])
dist_coeffs = np.matrix([1, 1, 1, 1, 1])

# Constants
INCH = 2.54 # Inches in cm

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Camera cannot be opened. Exiting...")
    exit()

detector = TagDetector()
line_detector = LineDetector()

while True:
    ret, frame = cap.read()
 
    if not ret:
        print("Stream cannot be opened.")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    edges = cv.Canny(gray, 50, 150, apertureSize = 3)

    ids, markers_corners = detector.detect(frame)

    if len(markers_corners) == 0:
        cv.imshow("HackIllinois", edges)
        if cv.waitKey(1) == ord("q"):
            break
        continue

    marker_corners = markers_corners[0][0]
    marker_side_vec = marker_corners[0] - marker_corners[1]

    pixels_per_cm = np.linalg.norm(marker_side_vec) / (6.5 * INCH)
    print(pixels_per_cm)

    pose = detector.estimate_pose(marker_corners, camera_mat, dist_coeffs)
    
    cv.aruco.drawDetectedMarkers(edges, markers_corners)

    cv.imshow("HackIllinois", edges)
    if cv.waitKey(1) == ord("q"):
        break

cap.release()
cv.destroyAllWindows()