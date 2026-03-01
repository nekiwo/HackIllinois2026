import cv2 as cv
import numpy as np
import configobj
import sys
from dxf_converter import DXFConverter
from tag_detector import TagDetector
from line_detector import LineDetector
from circle_detector import CircleDetector
from shape_simplifier import ShapeSimplifier

# Config
DEBUG = True
TAG_PERCENT = 0.20
PADDING_PERCENT = 0.03
EXPORT_FILENAME = "test.dxf"
# Constants
INCH = 25.4 # Inches in mm

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
line_detector = LineDetector(int(config["canny_th1"]), int(config["canny_th2"]), int(config["canny_aperture"]))
circle_detector = CircleDetector()
dxf_converter = DXFConverter()
shape_simplifier = ShapeSimplifier(int(config["simplify_length_threshold"]), int(config["simplify_dist_threshold"]), int(config["circle_clean_threshold"]))

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
subsample_percent = float(config["subsample_percent"])
line_subsample_percent = float(config["line_subsample_percent"])
contrast_multiplier = float(config["contrast_multiplier"])
brightness_coeff = float(config["brightness_coeff"])

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
    subsampled = cv.resize(frame, (int(width * subsample_percent), int(height * subsample_percent)), interpolation=cv.INTER_AREA)
    cv.imshow(name, subsampled)

def pipeline(frame):
    ids, markers_corners = detector.detect(frame)
    if len(markers_corners) == 0:
        show_image(frame)
        return [], []

    print("Detected Tag \n")
    marker_corners = markers_corners[0][0]
    marker_side_vec = marker_corners[0] - marker_corners[1]

    pixels_per_mm = np.linalg.norm(marker_side_vec) / (6.5 * INCH)

    frame = cv.undistort(frame, camera_mat, dist_coeffs)

    flatten_transform = cv.getAffineTransform(marker_corners[:-1], np.matrix([
        [tag_pixels + tag_padding_pixels, tag_pixels + tag_padding_pixels], # Lower left
        [tag_padding_pixels, tag_pixels + tag_padding_pixels], # Lower right
        [tag_padding_pixels, tag_padding_pixels] # Upper right
    ], dtype = np.float32))
    frame = cv.warpAffine(frame, flatten_transform, (height, width))
    frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)

    print("Frame Transformations \n")

    frame = frame * contrast_multiplier + brightness_coeff
    frame = cv.normalize(frame, frame, 0, 255, cv.NORM_MINMAX)
    frame = frame.astype(np.uint8)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # frame = cv.Canny(gray, int(config["canny_th1"]), int(config["canny_th2"]), L2gradient=True, apertureSize=int(config["canny_aperture"]))
    lines = line_detector.detect(gray, line_subsample_percent)
    circles = circle_detector.detect(gray)

    lines = shape_simplifier.simplify(lines)
    lines, circles = shape_simplifier.remove_apriltag(lines, circles, width - tag_pixels - tag_padding_pixels * 5, tag_pixels + tag_padding_pixels * 5)
    lines = shape_simplifier.clean_circles(lines, circles)

    if DEBUG:
        frame = line_detector.draw_lines(frame, lines)
        frame = circle_detector.draw(frame, circles)
        frame = cv.aruco.drawDetectedMarkers(frame, markers_corners)

    show_image(frame)
    return lines, circles

if not is_image:
    while True:
        ret, frame = cap.read()
    
        if not ret:
            print("Stream cannot be opened.")
            break

        lines, circles = pipeline(frame)

        key = cv.waitKey(1)
        if key == ord("s"):
            if len(lines) > 0 or len(circles) > 0:
                print("saving file...")
                dxf_converter.convert(lines, circles, EXPORT_FILENAME)
        elif key == ord("q"):
            print("closing stream...")
            break
else:
    frame = cv.imread(file_stream)
    lines, circles = pipeline(frame)
    dxf_converter.convert(lines, circles, EXPORT_FILENAME)
    while cv.waitKey(1) != ord("q"):
        continue

cap.release()
cv.destroyAllWindows()