import os
import cv2
import numpy as np

def GroundTruthtoTupleList(filename):
    filename_img = "image_1000/" + filename
    filename_txt = "txt_1000/" + filename.split(".jpg")[0] + ".txt"
    img = cv2.imread(filename_img)
    l = list()

    with open(filename_txt, 'r') as f:
        data = f.readlines()
        for line in data :
            position = line.split(",")[0 : 8]
            position = [int(float(position[i])) for i in range(8)]
            X = [position[0], position[2], position[4], position[6]]
            Y = [position[1], position[3], position[5], position[7]]

            cnt = np.array([[X[0], Y[0]], [X[1], Y[1]], [X[2], Y[2]], [X[3], Y[3]]])
            [[X_center, Y_center], [h, w], angle] = cv2.minAreaRect(cnt)

            if h > w :
                h, w = w, h
                angle += 90

            if angle < -45.0 :
                angle += 180

            t = (X_center, Y_center, h, w, angle)
            l.append(t)
    return img, l

if __name__ == "__main__":
    for filename in os.listdir('image_1000/'):
        img, lable = GroundTruthtoTupleList(filename)
