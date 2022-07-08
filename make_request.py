import sys
import matplotlib.pyplot as plt
import base64
import requests
from PIL import Image
import io

img = sys.argv[1]
with open(img, "rb") as img_file:
    img = base64.b64encode(img_file.read()).decode("utf-8")

url = "http://127.0.0.1:5000/api/recognize"
x = requests.post(url, json=dict(img=img)).json()
print("Decode:", x["predicted"], x['predicted_all'])
