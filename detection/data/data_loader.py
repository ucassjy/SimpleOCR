import os
import cv2
import numpy as np

def ResizeImage(img):
    """ Resize image to HMAX = 600 and WMAX = 2000. """

    img_shape = img.shape
    img_min = np.min(img_shape[0 : 2])
    img_max = np.max(img_shape[0 : 2])
    img_scale = np.array([600, 2000]) / np.array([img_min, img_max])
    img_scale = np.min(img_scale)

    img = cv2.resize(img, None, fx=img_scale, fy=img_scale)
    return img, img_scale

def GroundTruthtoTupleList(filename):
    """
    Read an img file and its txt file.
    Turn each ground truth to tuple (x, y, h, w, theta).
        x, y : geometric center of the bounding box
           h : short side of the bounding box
           w : long side of the bounding box
       theta : [-pi / 4, 3 * pi / 4)
    """

    filename_img = filename
    filename_txt = filename.replace("image", "txt", 1).split(".jpg")[0] + ".txt"
    img = cv2.imread(filename_img)
    if img is None :
        return None, None
    l = list()

    with open(filename_txt, 'r') as f:
        data = f.readlines()
        for line in data :
            position = line.split(",")[0 : 8]
            position = [int(float(position[i])) for i in range(8)]
            X = [position[0], position[2], position[4], position[6]]
            Y = [position[1], position[3], position[5], position[7]]

            cnt = np.array([[X[0], Y[0]], [X[1], Y[1]], [X[2], Y[2]], [X[3], Y[3]]])
            [[X_center, Y_center], [w, h], angle] = cv2.minAreaRect(cnt)

            angle = abs(angle)

            if h > w :
                h, w = w, h
                angle += 90.0

            if angle > 135.0:
                angle -= 180.0

            t = (X_center, Y_center, h, w, angle)
            l.append(t)
    return img, l

def GetBlobs(dirname):
    """
    Given a dirname and return blobs of the files in the directory.
    'Blobs' is a dict contains imagedata, groundtruth and imageinfo.
    """

    max_img_num = 10
    blobs = []

    i = 0
    for filename in os.listdir(dirname):
        img, l = GroundTruthtoTupleList(dirname + filename)
        if img is None :
            continue
        img, s = ResizeImage(img)
        img = img.reshape(1, img.shape[0], img.shape[1], 3)
        l = [(li[0]*s, li[1]*s, li[2]*s, li[3]*s, li[4]) for li in l]
        blobs.append({'name':filename, 'data': img, 'gt_list': l, 'im_info': np.array([img.shape[1], img.shape[2], s])}) #img.shape[1] = h, image.shape[2] = w
        i = i + 1

        if i == max_img_num:
            break

    return blobs

if __name__ == "__main__":
    blobs = GetBlobs('../../image_1000/')
    print(blobs)
    # cv2.imshow('img', blobs[0]['data'])
    # cv2.waitKey(0)
