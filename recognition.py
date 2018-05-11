import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

for filename in os.listdir('rewrite/'):
    filename_new = "rewrite/" + filename #"/home/h/a/hanlins/Desktop/OCR/image_1000/" + filename
    img = cv2.imread(filename_new)
    print(img.shape)
    if img is None:
        continue
