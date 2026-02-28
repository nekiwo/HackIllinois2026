import cv2 as cv
import numpy as np
import configobj
import sys
from tag_detector import TagDetector
from line_detector import LineDetector

# Config
DEBUG = True
# Constants
INCH = 2.54 # Inches in cm

if len(sys.argv) == 1:
    print("First argument must be config path. Exiting...")
    exit()

config = configobj.ConfigObj(sys.argv[1])
detector = TagDetector()
line_detector = LineDetector()

camera_mat = np.matrix([
    [config["f_x"], 0, config["c_x"]],
    [0, config["f_y"], config["c_y"]],
    [0, 0, 1]
], dtype=np.float32)
dist_coeffs = np.matrix([
    config["k_1"], config["k_2"], config["p_1"], config["p_2"], config["k_3"]
], dtype = np.float32)
width = int(config["width"])
height = int(config["height"])

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Camera cannot be opened. Exiting...")
    exit()

def show_image(frame):
    cv.imshow("HackIllinois", frame)
    if cv.waitKey(1) == ord("q"):
        return True
    return False

while True:
    ret, frame = cap.read()
 
    if not ret:
        print("Stream cannot be opened.")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    edges = cv.Canny(gray, 50, 150, apertureSize = 3)

    ids, markers_corners = detector.detect(frame)

    if len(markers_corners) == 0:
        if show_image(frame):
            break;
        continue

    marker_corners = markers_corners[0][0]
    marker_side_vec = marker_corners[0] - marker_corners[1]

    pixels_per_cm = np.linalg.norm(marker_side_vec) / (6.5 * INCH)

    flatten_transform = cv.getAffineTransform(marker_corners[:-1], np.matrix([
        [width / 2 - 50, height / 2 + 50],
        [width / 2 + 50, height / 2 + 50],
        [width / 2 + 50, height / 2 - 50]
    ], dtype = np.float32))
    frame = cv.warpAffine(frame, flatten_transform, (width, height))

    if DEBUG:
        cv.aruco.drawDetectedMarkers(frame, markers_corners)

    if show_image(frame):
        break;

cap.release()
cv.destroyAllWindows()