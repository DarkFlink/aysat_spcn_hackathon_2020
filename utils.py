from urllib import request
from urllib.request import urlopen
import requests
import albumentations as A
import numpy as np
import os
import cv2

def download(url, name): # doesn't work for gdocs
    if os.path.isfile('./' + name):
        return

    # download file
    dir = './'
    filename = os.path.join(dir, name)
    if not os.path.isfile(filename):
        response = urlopen(url)
        CHUNK = 16 * 1024
        with open(filename, 'wb') as f:
            while True:
                chunk = response.read(CHUNK)
                if not chunk:
                    break
                f.write(chunk)


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


def download_from_gdocs(url, filename):
    id = url[(url.find('id=') + len('id=')):]
    download_file_from_google_drive(id, filename)


def augmentation(img_mas):
    rotate = A.ShiftScaleRotate(p=0.5)
    scale = A.ShiftScaleRotate(shift_limit=0, scale_limit=0.9, rotate_limit=0, p=1)
    for img in img_mas:
        img_mas.append(scale(image=img)['image'])
        img_mas.append(rotate(image=img)['image'])
    return img_mas


def jpeg_from_mp4(path, destination, frame_ind = 1):
    cam = cv2.VideoCapture(path)
    try:
        # creating a folder named data
        if not os.path.exists(destination):
            os.makedirs(destination)
        # if not created then raise error
    except OSError:
        print('Error: Creating directory of data')
    # frame
    currentframe = 0
    frame_id = 0
    while (True):
        # reading from frame
        ret, frame = cam.read()

        if ret:
            if currentframe % frame_ind == 0:
                name = './data/frame' + str(frame_id) + '.jpg'
                print('Creating...' + name)
                cv2.imwrite(name, frame)
                frame_id += 1
            currentframe += 1
        else:
            break

    # Release all space and windows once done
    cam.release()
