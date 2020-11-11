from urllib.request import urlopen
from os.path import isfile, join
from urllib import request
import albumentations as A
import PIL.Image
from os import listdir
import numpy as np
import requests
import json
import cv2
import os
import glob
tiles = {0:"right", 1:"left", 2:"straight", 3:"three_cross", 4:"four_cross", 5:"empty"}
def download(url, name): # doesn't work for gdocs
    if os.path.isfile('./' + name):
        return

    # download file
    dir = '../'
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
    res = img_mas.copy()
    rotate = A.augmentations.transforms.Rotate(limit=30, p=1)
    el_t = A.augmentations.transforms.ElasticTransform(alpha=10, sigma=20, alpha_affine=50,p=1)
    for img in img_mas:
        res.append(rotate(image=img)['image'])
        res.append(el_t(image=img)['image'])
    return res


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
                name = f'./{destination}/frame_' + str(frame_id) + '_.jpg'
                print('Creating...' + name)
                new_frame = preproc(frame)
                cv2.imwrite(name, new_frame)
                frame_id += 1
            currentframe += 1
        else:
            break

    # Release all space and windows once done
    cam.release()


def preproc(src_img):
    img = src_img.copy()
    img = cv2.medianBlur(img, 5)
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(img,100,200)
    return np.asarray(canny)


def load_images(path = './data'):
    images_paths = []
    dir_tree = os.walk(path)
    folder = []
    valid_images = [".jpg", ".png", ".jpeg"]
    for el in dir_tree:
        if el[0] is path:
            continue
        folder.append(el)
    for address, dirs, files in folder:
        for file in files:
            for form in valid_images:
                if form in file:
                    images_paths.append(address + '/' + file)
    return load_images_from_path(images_paths)


def load_images_from_path(imgs_with_paths):
    images = []
    print(imgs_with_paths)
    for img in imgs_with_paths:
        img = cv2.imread(img)
        images.append(cv2.resize(img, (224, 224)))
    return images


def process_video(url, path, skip_rate=9, filename='data.mp4'):
    download(url, filename)
    jpeg_from_mp4(filename, path, skip_rate)



def get_train_x_y(_dir_path='./data'):
    x_data, y_data = [], []
    for dir_name in listdir(_dir_path):
        dir_path = f'{_dir_path}/{dir_name}'
        onlyfiles = [f for f in listdir(dir_path) if isfile(join(dir_path, f)) and '.json' in f]
        for filename in onlyfiles:
            with open(f'{_dir_path}/{dir_name}/{filename}') as file:
                data_json = json.load(file)
            for element in data_json:
                key = list(element.keys())[0]
                label = element[key]
                y_data.append(label)
             #   img = PIL.Image.open(key)
             #   x_data.append(np.asarray(img))
                x_data.append(np.asarray(load_images_from_path(['.'+key])[0]))
    return np.asarray(x_data), np.asarray(y_data)


def get_class(number):
    return tiles[number]

