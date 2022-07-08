import cv2
from flask import jsonify


def make_response(data={}, status=200):
    """
        - Make a resionable response with header
        - status default is 200 mean ok
    """
    res = jsonify(data)
    res.headers.add("Content-Type", "application/json")
    res.headers.add("Accept", "content-type/png")
    res.headers.add("Accept", "content-type/jpg")
    return res


def thresholding(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY_INV)
    return thresh
