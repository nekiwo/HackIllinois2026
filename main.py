import cv2 as cv
import numpy as n

from tag_detector import TagDetector
from line_detector import LineDetector

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

    edges = cv.Canny(gray,50,150,apertureSize = 3)
    line_detector.detect(frame, edges)

    #ids, marker_corners = detector.detect(frame)
    #print(ids)
    
    cv.imshow("HackIllinois", frame)
    #cv.aruco.drawDetectedMarkers(frame, marker_corners)

    #cv.imshow("HackIllinois", frame)
    if cv.waitKey(1) == ord("q"):
        break

cap.release()
cv.destroyAllWindows()