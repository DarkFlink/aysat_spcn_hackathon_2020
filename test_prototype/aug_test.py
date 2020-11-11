import cv2
from utils import augmentation
import numpy as np
import albumentations as A
img_mas =[cv2.imread('orig_2019-12-07-15-24-05/1575732273.462106943.jpg'), cv2.imread('orig_2019-12-07-15-24-05/1575732273.566128969.jpg'), cv2.imread('orig_2019-12-07-15-24-05/1575732273.708578109.jpg')]
#print(img_mas)
res = augmentation(img_mas)
for i in res:
    cv2.imshow('i',i)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
