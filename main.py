import cv2 as cv
import numpy as np
import configobj
import sys
from tag_detector import TagDetector
from line_detector import LineDetector

# Config
DEBUG = True
TAG_PIXELS = 100
TAG_PADDING_PIXELS = 25
# Constants
INCH = 2.54 # Inches in cm

config_path = None
file_stream = None
if len(sys.argv) > 2:
    for arg_i in range(len(sys.argv) - 1):
        arg = sys.argv[arg_i]
        arg_next = sys.argv[arg_i + 1]
        if arg == "--config" or arg == "-c":
            config_path = arg_next
        elif arg == "--file-stream" or arg == "-f":
            file_stream = arg_next


if config_path == None:
    print("Specify the `--config` flag. Exiting...")
    exit()

config = configobj.ConfigObj(config_path)
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

cap = None
if file_stream == None:
    cap = cv.VideoCapture(0)
else:
    cap = cv.VideoCapture(file_stream)

if cap == None or not cap.isOpened():
    print("Error starting stream. Exiting...")
    exit()

def show_image(frame, name="HackIllinois"):
    cv.imshow(name, frame)
    if cv.waitKey(1) == ord("q"):
        return True
    return False

while True:
    ret, frame = cap.read()
 
    if not ret:
        print("Stream cannot be opened.")
        break

    ids, markers_corners = detector.detect(frame)

    if len(markers_corners) == 0:
        if show_image(frame):
            break
        continue

    marker_corners = markers_corners[0][0]
    marker_side_vec = marker_corners[0] - marker_corners[1]

    pixels_per_cm = np.linalg.norm(marker_side_vec) / (6.5 * INCH)

    frame = cv.undistort(frame, camera_mat, dist_coeffs)

    flatten_transform = cv.getAffineTransform(marker_corners[:-1], np.matrix([
        [TAG_PIXELS + TAG_PADDING_PIXELS, TAG_PIXELS + TAG_PADDING_PIXELS], # Lower left
        [TAG_PADDING_PIXELS, TAG_PIXELS + TAG_PADDING_PIXELS], # Lower right
        [TAG_PADDING_PIXELS, TAG_PADDING_PIXELS] # Upper right
    ], dtype = np.float32))
    frame = cv.warpAffine(frame, flatten_transform, (height, width))
    frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # edges = cv.Canny(gray, 50, 150, apertureSize = 3)
    lines = line_detector.detect(frame, gray)


    if DEBUG:
        cv.aruco.drawDetectedMarkers(lines, markers_corners)
        if lines is not None:
            if show_image(lines):
                break
            continue

    if show_image(frame):
        break

cap.release()
cv.destroyAllWindows()