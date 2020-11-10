from utils import download_from_gdocs
from zipfile import ZipFile
import os

url = 'https://drive.google.com/u/0/uc?export=download&confirm=DCwi&id=1aIBD_eoIVcCVl8bjXt-oi6A4s-e-m5a4'
filename = 'dec7.zip'

download_from_gdocs(url, filename)

with ZipFile(filename, 'r') as zipObj:
    path_name = './data'
    os.mkdir(path_name)
    zipObj.extractall(path_name)