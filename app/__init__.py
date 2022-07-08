import cv2
import sys
from PIL import Image
from flask import Flask

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from vietocr.model.trainer import Trainer

config = Cfg.load_config_from_name("vgg_transformer")

config['weights'] = 'weights/transformerocr.pth'
config["cnn"]["pretrained"] = False
config["device"] = "cpu"
config["predictor"]["beamsearch"] = False

detector = Predictor(config)

app = Flask(__name__)

from app.routes import *
from app.utils import *
