import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

step = 0
for filename in os.listdir('/home/h/a/hanlins/Desktop/OCR/image_1000/'):
#for i in range(1):
#    filename = "TB1oAXWLXXXXXXoXXXXunYpLFXX.jpg"
    filename_new = "/home/h/a/hanlins/Desktop/OCR/image_1000/" +  filename
    img = cv2.imread(filename_new, cv2.IMREAD_GRAYSCALE)
#    ret, img = cv2.threshold(img,12,255,cv2.THRESH_BINARY)
    step += 1
    if (step>20):
        exit()

    #img = np.array(img)
    if img is None:
        continue

    filename_suf = filename.split(".jpg")[0]

    filename_full ="/home/h/a/hanlins/Desktop/OCR/txt_1000/" + filename_suf + ".txt"
    with open(filename_full,'r') as f:
#    with open("/home/h/a/hanlins/Desktop/OCR/txt_1000/TB1oAXWLXXXXXXoXXXXunYpLFXX.txt",'r') as f:
        data = f.readlines()  # The txt file of 1 picture.
        Num = 0
        Err = 0
        for line in data:
            position = line.split(",")[0:8]
            label = line.split(",")[8]

            # print(type(position[0]))
            for i in range(len(position)):
                position[i] = int(float(position[i]))
            X = [position[0],position[2],position[4],position[6]]
            Y = [position[1],position[3],position[5],position[7]]

            cnt = np.array([[position[0], position[1]],[position[2], position[3]],[position[4], position[5]],[position[6], position[7]]])
            rect = cv2.minAreaRect(cnt)
            angle = rect[2]
            box = np.int0(cv2.boxPoints(rect))
            # if(angle!=-90.0 and angle != 0):
            #     cv2.drawContours(img,[box],-1,(0,255,0),3)
            #     cv2.imshow("Image",img)
            #     cv2.waitKey(0)
            #

            X_centre = rect[0][0]
            Y_centre = rect[0][1]
            #
            # if (filename_suf=="TB1..FLLXXXXXbCXpXXunYpLFXX" and Num == 13):
            #     cv2.drawContours(img,[box],-1,(0,255,0),3)
            #     cv2.imshow("Image",img)
            #     cv2.waitKey(0)
            #     exit()


            if (angle==-90.0 or angle==0):
                X_start = min(X)
                X_end = max(X)
                Y_start = min(Y)
                Y_end = max(Y)
                crop_img = img[Y_start:Y_end,X_start:X_end]
                plt.imshow(crop_img,cmap="gray")
                print(label,Num)
                rewrite_name = "/home/h/a/hanlins/Desktop/OCR/rewrite/" + filename_suf + "_" + str(Num) + ".jpg"
                plt.savefig(rewrite_name)
                Num += 1

            else:
                Y_MAX = np.shape(img)[0]
                X_MAX = np.shape(img)[1]
                M = np.float32([[1,0,X_MAX],[0,1,Y_MAX]])
                img_new = cv2.warpAffine(img,M,(Y_MAX*3, X_MAX*3))
                RECT = (rect[0][0] + X_MAX,rect[0][1] + Y_MAX)
                M = cv2.getRotationMatrix2D(RECT, angle, 1)
            #    if(-angle > 45):
            #        X_MAX, Y_MAX = Y_MAX, X_MAX
                img_new = cv2.warpAffine(img_new, M, (Y_MAX*3, X_MAX*3))
                #img_new = img
                print(X_MAX, Y_MAX)
            #    plt.imshow(img_new)
                # print(angle)
                # print(label)
            #    plt.imshow(img_new)
            #    plt.show()
            #    exit()
                X_start = int(X_centre - rect[1][0]/2 + X_MAX)
                X_end = int(X_centre + rect[1][0]/2 + X_MAX)
                Y_start = int(Y_centre - rect[1][1]/2 + Y_MAX)
                Y_end = int(Y_centre + rect[1][1]/2 + Y_MAX)
                print(X_start, X_end, Y_start, Y_end)
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

                # if (filename_suf == "TB1oAXWLXXXXXXoXXXXunYpLFXX" and Num==7):
                #     print(X_start, X_end, Y_start, Y_end)
                #     crop_img = img_new[Y_start:Y_end,X_start:X_end]
                #     plt.imshow(img_new)
                #     plt.show()
                    #
                    # cv2.drawContours(img,[box],-1,(0,255,0),3)
                    # cv2.imshow("Image",img)
                    # cv2.waitKey(0)
                    #
                    # exit()




                print(X_start, X_end, Y_start, Y_end)
                # if (filename_suf=="TB1..FLLXXXXXbCXpXXunYpLFXX" and Num == 13):
                #     print(X_start, X_end, Y_start, Y_end)
                #     crop_img = img_new[Y_start:Y_end,X_start:X_end]
                #     plt.imshow(crop_img)
                #     plt.show()
                #     exit()

                crop_img = img_new[Y_start:Y_end,X_start:X_end]
                plt.imshow(crop_img,cmap="gray")
                print(label,Num)
                rewrite_name = "/home/h/a/hanlins/Desktop/OCR/rewrite/" + filename_suf + "_" + str(Num) + ".jpg"
                f = open("/home/h/a/hanlins/Desktop/OCR/target_label.txt",'a')
                f.write(rewrite_name + ',' + label)

                f.close()
                plt.savefig(rewrite_name)
                Num += 1





            # X = np.array(X)
            # Y = np.array(Y)
            # print(type(X))
            # exit()
#            X = [np.min(X_mid),X_mid[np.where(np.max(Y_mid))],np.max(X_mid),X_mid[np.where(np.min(Y_mid))]]


            # if (Y[1] == Y[2]):
            #     X_start = min(X[1],X[2])
            #     X_end = max(X[1],X[2])
            #     Y_start = min(Y[2],Y[3])
            #     Y_end = max(Y[2],Y[3])
            #     crop_img = img[Y_start:Y_end,X_start:X_end]
            #     plt.imshow(crop_img)
            #     rewrite_name = "/home/h/a/hanlins/Desktop/OCR/rewrite/" + filename_suf + "_" + str(Num) + ".jpg"
            #     plt.savefig(rewrite_name)
            #     Num += 1
            # else:
            #     X_centre = min(X)
            #     Y_centre = Y[X.index(X_centre)]
            #     if ((max(Y)-Y[X.index(min(X))])==0):
            #         print(filename)
            #         print(label)
            #
