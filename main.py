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

config_path = "sony_a6000_photo.cfg"
file_stream = None

if __name__ == "__main__":
    argc = len(sys.argv)
    if argc > 2:
        for arg_i in range(argc):
            arg = sys.argv[arg_i]
            if arg_i <= argc - 2:
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

if __name__ == "__main__":
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

def pipeline(frame, show=True):
    ids, markers_corners = detector.detect(frame)
    if show:
        if len(markers_corners) == 0:
            show_image(frame)
            return [], []

    print("Detected Tag \n")
    marker_corners = markers_corners[0][0]

    frame = cv.undistort(frame, camera_mat, dist_coeffs)

    flatten_transform = cv.getAffineTransform(marker_corners[:-1], np.matrix([
        [tag_pixels + tag_padding_pixels, tag_pixels + tag_padding_pixels], # Lower left
        [tag_padding_pixels, tag_pixels + tag_padding_pixels], # Lower right
        [tag_padding_pixels, tag_padding_pixels] # Upper right
    ], dtype = np.float32))
    frame = cv.warpAffine(frame, flatten_transform, (height, width))
    frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)

    pixels_per_mm = tag_pixels / (0.25 * 6.5 * INCH)

    print("Frame Transformations \n")

    frame = frame * contrast_multiplier + brightness_coeff
    frame = cv.normalize(frame, frame, 0, 255, cv.NORM_MINMAX)
    frame = frame.astype(np.uint8)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # frame = cv.Canny(gray, int(config["canny_th1"]), int(config["canny_th2"]), L2gradient=True, apertureSize=int(config["canny_aperture"]))
    lines = line_detector.detect(gray, line_subsample_percent)
    circles = circle_detector.detect(gray)

    lines, circles = shape_simplifier.remove_apriltag(lines, circles, width - tag_pixels - tag_padding_pixels * 3, tag_pixels + tag_padding_pixels * 3)
    lines = shape_simplifier.simplify(lines)
    lines = shape_simplifier.clean_circles(lines, circles)

    if DEBUG:
        frame = line_detector.draw_lines(frame, lines)
        frame = circle_detector.draw(frame, circles)
        # frame = cv.aruco.drawDetectedMarkers(frame, markers_corners)

    if show:
        show_image(frame)

    if lines is not None and len(lines) > 0:
        lines = np.array(lines) * (1.0 / pixels_per_mm)
    if circles is not None and len(circles) > 0:
        circles = np.array(circles) * (1.0 / pixels_per_mm)

    return lines, circles

def start_standalone():
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

if __name__ == "__main__":
    start_standalone()