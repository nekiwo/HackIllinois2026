from flask import Flask, send_from_directory, request, make_response
from main import pipeline
import cv2 as cv
from dxf_converter import DXFConverter

app = Flask(__name__)

@app.route("/")
def landing_page():
    return send_from_directory("static", "index.html")

@app.route("/index.js")
def landing_page_script():
    return send_from_directory("static", "index.js")

@app.route("/index.css")
def landing_page_style():
    return send_from_directory("static", "index.css")

@app.route("/upload", methods = ["POST"])
def upload_photo():
    if request.method == "POST":
        photo = request.files["file"]
        photo.save("uploads/" + photo.filename)
        resp = make_response(send_from_directory("static", "uploaded.html"))
        resp.set_cookie("photo", photo.filename)

        return resp

dxf_converter = DXFConverter()
@app.route("/get_photo/<photo>", )
def get_photo(photo):
    frame = cv.imread("uploads/" + photo)
    lines, circles = pipeline(frame, False)
    dxf_name = photo.replace(".", "") + ".dxf"
    dxf_converter.convert(lines, circles, "uploads/out/" + dxf_name)
    print("replying...")
    return send_from_directory("uploads/out", dxf_name)
    
app.run(debug = True)