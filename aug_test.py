import cv2
from utils import augmentation
import numpy as np
import albumentations as A
img_mas =[cv2.imread('data/1575732377.955286979.jpg'), cv2.imread('data/1575732378.060125112.jpg'), cv2.imread('data/1575732378.166352033.jpg')]
#print(img_mas)
res = augmentation(img_mas)
#print(res)
#cv2.imshow('test',img)
#print(type(img))
#for i in res:
cv2.imshow('i',res[4])
cv2.waitKey(0)
##res = augmentation(img)
# closing all open windows
cv2.destroyAllWindows()
