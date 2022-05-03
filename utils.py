from types import FunctionType
import cv2
from monads import Option
import numpy as np
from urllib.request import urlopen
from subprocess import check_output
import json



def make_request(url):
    
    parse_url = lambda url: f'"{url}"'
    cmd = f"curl {parse_url(url)}"
    response = check_output(cmd).decode()
    return json.loads(response)


def url_to_image(url, readFlag=cv2.IMREAD_COLOR, width=600):

    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, readFlag)

    return image


def load_image_from_path(p):
    if "http" in p[:6]:
        return Option(lambda : url_to_image(p))
    return Option(lambda : cv2.imread(p))


def predict_given_path(p:str, func:FunctionType):
    if "http" in p[:6]:
        return Option(lambda : url_to_image(p)).apply(func)
    return Option(lambda : cv2.imread(p)).apply(func)
