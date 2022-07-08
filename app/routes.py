import numpy as np
from app import app, detector
from .utils import make_response, thresholding
from flask import request, jsonify
from flask_cors import cross_origin
import cv2
import base64
from io import BytesIO
from PIL import Image


@app.route("/", methods=["GET"])
@cross_origin()
def index():
    return make_response(dict(greeting="Hello"))


@app.route("/api/recognize", methods=["POST"])
@cross_origin()
def recognize():

    data = request.get_json()
    file = data.get("img")
    img = Image.open(BytesIO(base64.b64decode(file)))

    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    thresh_img = thresholding(img)

    # dilation
    kernel = np.ones((3, 85), np.uint8)
    dilated = cv2.dilate(thresh_img, kernel, iterations=1)

    contours, *_ = cv2.findContours(
        dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    sorted_contours_lines = sorted(
        contours, key=lambda ctr: cv2.boundingRect(ctr)[1]
    )  # (x, y, w, h)

    img_shape = Image.fromarray(img)
    predicted_all = detector.predict(img_shape)

    predicted = []

    for ctr in sorted_contours_lines:

        x, y, w, h = cv2.boundingRect(ctr)
        img_shape = img[y : y + h]
        img_shape = Image.fromarray(img_shape)
        s = detector.predict(img_shape)
        predicted.append(s)

    predicted = "\n".join(predicted)

    return make_response(data=dict(predicted=predicted, predicted_all=predicted_all))
