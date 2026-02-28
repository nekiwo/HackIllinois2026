import cv2 as cv
import numpy as np
import configobj
import sys
from tag_detector import TagDetector
from line_detector import LineDetector
import time

# Config
DEBUG = True
TAG_PERCENT = 0.20
PADDING_PERCENT = 0.03
SUBSAMPLE_PERCENT = 1.0
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
tag_pixels = int(width * TAG_PERCENT)
tag_padding_pixels = int(width * PADDING_PERCENT)

cap = None
is_image = False
if file_stream == None:
    cap = cv.VideoCapture(0)
else:
    if file_stream.find(".jpg") != -1 or file_stream.find(".png") != -1:
        is_image = True
    else:
        cap = cv.VideoCapture(file_stream)

if not is_image and (cap == None or not cap.isOpened()):
    print("Error starting stream. Exiting...")
    exit()

def show_image(frame, name="HackIllinois"):
    subsampled = cv.resize(frame, (int(width * SUBSAMPLE_PERCENT), int(height * SUBSAMPLE_PERCENT)), interpolation=cv.INTER_AREA)
    cv.imshow(name, subsampled)

def pipeline(frame):
    ids, markers_corners = detector.detect(frame)

    if len(markers_corners) == 0:
        show_image(frame)
        return

    marker_corners = markers_corners[0][0]
    marker_side_vec = marker_corners[0] - marker_corners[1]

    pixels_per_cm = np.linalg.norm(marker_side_vec) / (6.5 * INCH)

    frame = cv.undistort(frame, camera_mat, dist_coeffs)

    flatten_transform = cv.getAffineTransform(marker_corners[:-1], np.matrix([
        [tag_pixels + tag_padding_pixels, tag_pixels + tag_padding_pixels], # Lower left
        [tag_padding_pixels, tag_pixels + tag_padding_pixels], # Lower right
        [tag_padding_pixels, tag_padding_pixels] # Upper right
    ], dtype = np.float32))
    frame = cv.warpAffine(frame, flatten_transform, (height, width))
    frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame = cv.Canny(gray, 50, 150, apertureSize = 3)
    lines = line_detector.detect(frame, gray)

    if DEBUG:
        if lines is not None:
            cv.aruco.drawDetectedMarkers(lines, markers_corners)
            show_image(lines)
            return

    show_image(frame)

if not is_image:
    while True:
        ret, frame = cap.read()
    
        if not ret:
            print("Stream cannot be opened.")
            break

        pipeline(frame)
        if cv.waitKey(1) == ord("q"):
            break
else:
    frame = cv.imread(file_stream)
    pipeline(frame)
    while cv.waitKey(1) != ord("q"):
        continue

cap.release()
cv.destroyAllWindows()