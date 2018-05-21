import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

step = 0
MAX = 0
Threshold = 32
MAX_LENGTH = 256
ff = open("target_label.txt",'w')

for filename in os.listdir('image_1000/'): #'/home/h/a/hanlins/Desktop/OCR/image_1000/'):
#for i in range(1):
#    filename = "TB1oAXWLXXXXXXoXXXXunYpLFXX.jpg"
    filename_new = "image_1000/" + filename #"/home/h/a/hanlins/Desktop/OCR/image_1000/" + filename

    img = cv2.imread(filename_new, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 20)
    #ret,img = cv2.threshold(img, 127,255,cv2.THRESH_TOZERO)
    step += 1
    print(step)
    if (step > 10):

        exit()

    #img = np.array(img)

    filename_suf = filename.split(".jpg")[0]

    filename_full = "txt_1000/" + filename_suf + ".txt" #"/home/h/a/hanlins/Desktop/OCR/txt_1000/" + filename_suf + ".txt"
    with open(filename_full, 'r') as f:
#    with open("/home/h/a/hanlins/Desktop/OCR/txt_1000/TB1oAXWLXXXXXXoXXXXunYpLFXX.txt",'r') as f:
        data = f.readlines()  # The txt file of 1 picture.
        Num = 0
        Err = 0
        for line in data:
            position = line.split(",")[0:8]
            label = line.split(",")[8]

            position = [int(float(position[i])) for i in range(8)]
            X = [position[0], position[2], position[4], position[6]]
            Y = [position[1], position[3], position[5], position[7]]

            cnt = np.array([[position[0], position[1]],[position[2], position[3]],[position[4], position[5]],[position[6], position[7]]])
            rect = cv2.minAreaRect(cnt)
            angle = rect[2]

            X_centre = rect[0][0]
            Y_centre = rect[0][1]

            if (angle == -90.0 or angle == 0.0):
                X_start = min(X)
                X_end = max(X)
                Y_start = min(Y)
                Y_end = max(Y)
                crop_img = img[Y_start:Y_end, X_start:X_end]
                if (crop_img.shape[1] < crop_img.shape[0]):
                    crop_img = np.rot90(crop_img)
                x_s = Threshold
                if crop_img.shape[0]==0:
                    continue
                y_s = int((crop_img.shape[1]/crop_img.shape[0]) * Threshold)
                crop_img = cv2.resize(crop_img,(y_s, x_s))
                rest = MAX_LENGTH - crop_img.shape[1]
                if rest < 0:
                    rest = 0
                stack = np.ones([Threshold, rest]) * 255
                crop_img = np.hstack((crop_img, stack))
            #    print(crop_img.shape)
            #    plt.imshow(crop_img, cmap="gray")
            #    print(label, Num, '0')

                rewrite_name = "rewrite/" + filename_suf + "_" + str(Num) + ".jpg" #"/home/h/a/hanlins/Desktop/OCR/rewrite/" + filename_suf + "_" + str(Num) + ".jpg"
                cv2.imwrite(rewrite_name,crop_img)
                ff.write(rewrite_name + '__' + label)
                Num += 1

            else:
                Y_MAX = np.shape(img)[0]
                X_MAX = np.shape(img)[1]
                M = np.float32([[1,0,X_MAX], [0,1,Y_MAX]])
                img_new = cv2.warpAffine(img, M, (Y_MAX*3, X_MAX*3))
                RECT = (rect[0][0] + X_MAX, rect[0][1] + Y_MAX)
                M = cv2.getRotationMatrix2D(RECT, angle, 1)
                img_new = cv2.warpAffine(img_new, M, (Y_MAX*3, X_MAX*3))

                X_start = int(X_centre - rect[1][0]/2 + X_MAX)
                X_end = int(X_centre + rect[1][0]/2 + X_MAX)
                Y_start = int(Y_centre - rect[1][1]/2 + Y_MAX)
                Y_end = int(Y_centre + rect[1][1]/2 + Y_MAX)
                crop_img = img_new[Y_start:Y_end, X_start:X_end]
                if (crop_img.shape[1] < crop_img.shape[0]):
                    crop_img = np.rot90(crop_img)
                x_s = Threshold
                if crop_img.shape[0]==0:
                    continue
                # y_s = int((crop_img.shape[1]/crop_img.shape[0]) * Threshold)
                y_s = int(MAX_LENGTH)
                crop_img = cv2.resize(crop_img,(y_s, x_s))
                print(crop_img.shape)
                # plt.imshow(crop_img)
                # plt.show()
                # exit()
                # rest = MAX_LENGTH - crop_img.shape[1]
                # if rest < 0:
                #     rest = 0
                # stack = np.ones([Threshold, rest]) * 255
                # crop_img = np.hstack((crop_img, stack))

                # if (X_start < 0):
                #     M = np.float32([[1,0,-X_start],[0,1,0]])
                #     img_new = cv2.warpAffine(img_new,M,(Y_MAX, X_MAX))
                #     X_end = X_end - X_start
                #     X_start = 0
                # if (Y_start < 0):
                #     M = np.float32([[1,0,0],[0,1,-Y_start]])
                #     img_new = cv2.warpAffine(img_new,M,(Y_MAX, X_MAX))
                #     Y_end = Y_end - Y_start
                #     Y_start = 0
                # if (X_end > X_MAX):
                #     M = np.float32([[1,0,X_MAX-X_end],[0,1,0]])
                #     img_new = cv2.warpAffine(img_new,M,(Y_MAX, X_MAX))
                #     X_start = X_start +X_MAX - X_end
                #     X_end = X_MAX
                # if (Y_end > Y_MAX):
                #     M = np.float32([[1,0,0],[0,1,Y_MAX-Y_end]])
                #     img_new = cv2.warpAffine(img_new,M,(Y_MAX, X_MAX))
                #     Y_start = Y_start +Y_MAX - Y_end
                #     Y_end = Y_MAX
                # crop_img1 = crop_img
                # if (crop_img.shape[0] > crop_img.shape[1]):
                #     crop_img = crop_img.T
                # if (crop_img1.shape[0] > Threshold):
                #     crop_img1 = cv2.resize(crop_img1, (Threshold, int(crop_img1.shape[1]*(Threshold/crop_img1.shape[0]))))
                # if (crop_img1.shape[1] > Threshold):
                #     crop_img1 = cv2.resize(crop_img1, (int(crop_img1.shape[0]*(Threshold/crop_img1.shape[1])),Threshold))
                # if crop_img1.shape[0] > MAX:
                #     MAX = crop_img1.shape[0]
                # if crop_img1.shape[1] > MAX:
                #     MAX = crop_img1.shape[1]
                # if(MAX == 2164):
                #     print(crop_img1.shape)
                #     print(crop_img.shape)
                #     exit()
                # print(MAX)
                rewrite_name = "rewrite/" + filename_suf + "_" + str(Num) + ".jpg" #"/home/h/a/hanlins/Desktop/OCR/rewrite/" + filename_suf + "_" + str(Num) + ".jpg"
                cv2.imwrite(rewrite_name,crop_img)
                ff.write(rewrite_name + '__' + label)
                Num += 1

ff.close()
