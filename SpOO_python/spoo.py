
# importing all the libraries

import pickle
import cv2
import imutils
import numpy as np
import time
from face_detection import FaceDetection
from scipy import signal
import sys
from numpy.linalg import inv
import dlib
import imutils
import time


frameCount = 1
final_sig = []
global gMask
global gBox


np.set_printoptions(threshold=sys.maxsize)

# TODO: Add face detection inside code only
# def face_detection(frame):
#     blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
#     	(300, 300), (104.0, 177.0, 123.0))
#     net.setInput(blob)
#     detections = net.forward()
#     for i in range(0, detections.shape[2]):
#     	# extract the confidence (i.e., probability) associated with the
#     	# prediction
#     	confidence = detections[0, 0, i, 2]
#
#     	# filter out weak detections by ensuring the `confidence` is
#     	# greater than the minimum confidence
#     	if confidence < 0.5:
#     		continue
#
#     	# compute the (x, y)-coordinates of the bounding box for the
#     	# object
#     	box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#     	(startX, startY, endX, endY) = box.astype("int")
#
#     	# draw the bounding box of the face along with the associated
#     	# probability
#     	text = "{:.2f}%".format(confidence * 100)
#     	y = startY - 10 if startY - 10 > 10 else startY + 10
#     	cv2.rectangle(frame, (startX, startY), (endX, endY),
#     		(0, 0, 255), 2)
#     	cv2.putText(frame, text, (startX, y),
#     		cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
#
#     # show the output frame
#     cv2.imshow("Frame", frame)
#     key = cv2.waitKey(1) & 0xFF
#     return 0
#     #


def dethreding(z):

    return signal.detrend(z)

    # T=len(z)
    # lamba=10
    # I  = np.identity(T)
    # filt = [1,-2,1]* np.ones((1,T-2),dtype=np.int).T
    # D2 = scipy.sparse.spdiags(data.T, (range(0,3)),T-2,T)
    # detrended_sig =
    # return z_stat


def face_detect_and_thresh(frame):
    # lower = np.array([0, 1,0], dtype = "uint8")
    # upper = np.array([179, 255, 255], dtype = "uint8")
    lower = np.array([0, 58, 50], dtype = "uint8")
    upper = np.array([30, 255, 255], dtype = "uint8")
    # for detecting skin lower and upper are used for creating a mask
    fd = FaceDetection()
    frame, face_frame, ROI1, ROI2, status, mask, extra, box = fd.face_detect(frame)
    frame = extra
    # cv2.imshow("face0",extra)
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)
    cv2.imshow("face0", converted)
    cv2.imshow("face3", skinMask)
    # apply a series of erosions and dilations to the mask
    # using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel, iterations=2)
    skinMask = cv2.dilate(skinMask, kernel, iterations=2)
    # blur the mask to help remove noise, then apply the
    # mask to the frame
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    skin = cv2.bitwise_and(frame, frame, mask=skinMask)
    cv2.imshow("skin2", skin)
    gMask = skinMask
    global gBox
    gBox = box
    # show the skin in the image along with the mask
    # cv2.imshow("images", np.hstack([frame, skin]))
    # cv2.imshow("mask",np.hstack([mask]))

    for _ in range(np.shape(mask)[0]):
        for j in range(np.shape(mask)[1]):
            if mask[_][j] == 1:
                mask[_][j] = 1
            else:
                mask[_][j] = 0
    mask = mask.astype(int)
    return skin, frame, mask


def spartialAverage(thresh, frame):
    a = list(np.argwhere(maskT == 1))
    # x=[i[0] for i in a]
    # y=[i[1] for i in a]
    # p=[x,y]
    ind_img = (np.vstack((a)))
    sig_fin = np.zeros([np.shape(ind_img)[0], 3])
    test_fin = []
    for i in range(np.shape(ind_img)[0]):
        sig_temp = frame[ind_img[i, 0], ind_img[i, 1], :]
        sig_temp = sig_temp.reshape((1, 3))
        if sig_temp.any() != 0:
            sig_fin = np.concatenate((sig_fin, sig_temp))
    # print(sig_fin)
    for _ in sig_fin:
        if sum(_) > 0:
            # test_fin=np.concatenate((test_fin,_))
            test_fin.append(_)
    # print("min=>")
    a = [item for item in sig_fin if sum(item) > 0]
    # print(min(a, key=sum))
    min_value = sum(min(a, key=sum))
    max_value = sum(max(a, key=sum))
    # print(sum1)
    img_rgb_mean = np.nanmean(test_fin, axis=0)
    print(img_rgb_mean)
    return img_rgb_mean, min_value, max_value


def MeanRGB(thresh, frame, img_rgb, last_stage, min_value, max_value):
    cv2.imshow("threshh", img_rgb)
    print("==<>>")
    # print(img_rgb)
    # cv2.waitKey()
    # print(img_rgb[0])
    # thresh=thresh.reshape((1,3))
    # img_rgb_mean=np.nanmean(thresh,axis=0)
    a = [item for item in img_rgb[0] if (sum(item) < max_value and sum(item) > 200)]
    # a = filter(lambda (x,y,z) : i+j+k>764 ,frame[0])
    # print(a[1:10])
    # img_temp = [item for item in img_rgb if sum(item)>764]
    # print(frame[0])
    # print(img_temp)
    # print(np.mean(a, axis=(0)))
    if a:
        print("==>")
        # print(a)
        print("==>")
        img_mean = np.mean(a, axis=(0))
        # print(img_mean)

        return img_mean[::-1]
    else:
        return last_stage


def MeanRGB2(thresh, frame, img_rgb, min_value, max_value):
    cv2.imshow("threshh", img_rgb)
    print("==<>>")
    # print(img_rgb)
    # cv2.waitKey()
    # print(img_rgb[0])
    # thresh=thresh.reshape((1,3))
    # img_rgb_mean=np.nanmean(thresh,axis=0)
    a = [item for item in img_rgb[0] if (sum(item) < max_value and sum(item) > min_value)]
    # a = filter(lambda (x,y,z) : i+j+k>764 ,frame[0])
    # print(a[1:10])
    # img_temp = [item for item in img_rgb if sum(item)>764]
    # print(frame[0])
    # print(img_temp)
    # print(np.mean(a, axis=(0)))

    print("==>")
    # print(a)
    print("==>")
    img_mean = np.mean(a, axis=(0))
    # print(img_mean)

    return img_mean[::-1]


# def MeanRGB(thresh, frame, img_rgb, last_stage, max_value, min_value):
#     cv2.imshow("threshh", img_rgb)
#     print("==<>>")
#     # print(img_rgb)
#     # cv2.waitKey()
#     # print(img_rgb[0])
#     # thresh=thresh.reshape((1,3))
#     # img_rgb_mean=np.nanmean(thresh,axis=0)
#     a = [item for item in img_rgb[0] if (sum(item) < 764 and sum(item) > 200)]
#     # a = filter(lambda (x,y,z) : i+j+k>764 ,frame[0])
#     # print(a[1:10])
#     # img_temp = [item for item in img_rgb if sum(item)>764]
#     # print(frame[0])
#     # print(img_temp)
#     # print(np.mean(a, axis=(0)))
#     if a:
#         print("==>")
#         # print(a)
#         print("==>")
#         img_mean = np.mean(a, axis=(0))
#         # print(img_mean)
#
#         return img_mean[::-1]
#     else:
#         return last_stage
#
#
# def MeanRGB2(thresh, frame, img_rgb):
#     cv2.imshow("threshh", img_rgb)
#     print("==<>>")
#     # print(img_rgb)
#     # cv2.waitKey()
#     # print(img_rgb[0])
#     # thresh=thresh.reshape((1,3))
#     # img_rgb_mean=np.nanmean(thresh,axis=0)
#     a = [item for item in img_rgb[0] if (sum(item) < 764 and sum(item) > 200)]
#     # 764 being the max value of skin in HSV range
#
#     # 200 being the min value of hsv range
#
#
#     # a = filter(lambda (x,y,z) : i+j+k>764 ,frame[0])
#     # print(a[1:10])
#     # img_temp = [item for item in img_rgb if sum(item)>764]
#     # print(frame[0])
#     # print(img_temp)
#     # print(np.mean(a, axis=(0)))
#
#     # print("==>")
#     # print(a)
#     # print("==>")
#     img_mean = np.mean(a, axis=(0))
#     # print(img_mean)
#
#     return img_mean[::-1]


def face_detect_and_thresh2(frame):
    lower = np.array([0, 58, 50] , dtype="uint8")
    upper = np.array([30, 255, 255], dtype="uint8")
    face_frame = frame[gBox[0]:gBox[1], gBox[2]:gBox[3]]
    # cv2.imshow("face_testing",face_frame)
    frame = face_frame
    cv2.imshow("testing", frame)
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)
    # apply a series of erosions and dilations to the mask
    # using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel, iterations=2)
    skinMask = cv2.dilate(skinMask, kernel, iterations=2)
    # blur the mask to help remove noise, then apply the
    # mask to the frame
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    skin = cv2.bitwise_and(frame, frame, mask=skinMask)
    cv2.imshow("skin", skin)
    # print(np.mean(skin, axis=(0, 1)))
    # cv2.imshow("face_testing",skin)
    return skin, frame


def preprocess(z1, z2, detrended_RGB, window_size, size_video, duration, frame):
    temp = (int(size_video/duration))
    f = frame-2

    main_R = []
    main_B = []
    out = []
    for i in range(len(detrended_RGB)-f+1):

        temp_R = z1[i:i+f-1]
        temp_B = z2[i:i+f-1]
        p = [list(a) for a in zip(temp_R, temp_B)]

        out.append(p)
        # if not main_R:
        #     main_R.append(temp_R)
        # else:
        #     main_R=[main_R,temp_R]
        #
        # if not main_B:
        #     main_B.append(temp_B)
        # else:
        #     main_B=[main_B,temp_B]

    # out=[main_R,main_B]
    # print(out[0])
    return out[0]


def SPooEsitmate(final_sig, video_size, frames, seconds):
    A = 101.6
    B = 5.834

    #  A and B are empirically determined coefficients, Iac and Idc are respectively the amplitudes of the pulsatile (ac) and dc components
    # the determination of these values were done on the basis of the paper published
    z1 = [item[0] for item in final_sig]
    z3 = [item[2] for item in final_sig]
    #The mean average value of Blue and Red  is taken from the set of frames
    SPO_pre = []
    for _ in range(len(z1)):
        SPO_pre.append([z1[_], z3[_]])
    Spo2 = preprocess(z1, z3, SPO_pre, 10, video_size, seconds, frames)

    R_temp = [item[0] for item in Spo2]
    DC_R_comp = np.nanmean(R_temp)
    AC_R_comp = np.std(R_temp)

    I_r = AC_R_comp/DC_R_comp

    B_temp = [item[1] for item in Spo2]
    DC_B_comp = np.nanmean(B_temp)
    AC_B_comp = np.std(B_temp)

    I_b = AC_B_comp/DC_B_comp
    SpO2_value = (A-B*((I_b*650)/(I_r*950)))
    # Valye 650 and 950 was chosen as the wavelengths of red and near infrared
    return SpO2_value


# main code begins from here and all the above helper fuctions are called


# Enter the path of video you wanna give into the code
cap = cv2.VideoCapture("videos/Corona_SPO2.avi")

# Calculates the total number of frames present in the video
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count/fps
seconds = duration % 60

print(length)

if (cap.isOpened() == False):
    print("Error opening video file")  # Catching anykind of error while opening video

while(cap.isOpened()):
    ret, frame = cap.read()

    # cv2.waitKey()
    if frameCount == 1:
        # cv2.imshow("testing if flipped",frame)  #just run this if you have issue with inverted frames, can result in fail detection
        # cv2.waitKey()
        # cv2.imshow("testing if flipped2",imutils.rotate_bound(frame, 90))  #just run this if you have issue with inverted frames, can result in fail detection
        # cv2.waitKey()
        # frame= imutils.rotate_bound(frame, 90)
        firstFrame = frame
        # The first frame is sent to the fuction to make a face mask
        thresh, img_rgb, maskT = face_detect_and_thresh(firstFrame)
        cv2.imshow("img_rgb", img_rgb)
        # cv2.waitKey()
        frameCount += 1

    if ret == True:
        # cv2.imshow("testing if flipped",frame)  #just run this if you have issue with inverted frames, can result in fail detection
        frameCount += 1
        # frame= imutils.rotate_bound(frame, 90)

        print(frameCount)
        cv2.imshow('Frame', frame)

        start = time.time()
        # rest of frames are sent to a function to have a threshold/overlap with mask
        thresh, img_rgb = face_detect_and_thresh2(frame)
        end = time.time()
        print(end - start)

        start = time.time()

        final_sig.append(spartialAverage(maskT,frame)) # The frames are then sent to spartialAverage to find RGB mean values
        end = time.time()
        print(end - start)

        cv2.waitKey()

        # start = time.time()
        # if final_sig:
        #     # The frames are then sent to spartialAverage to find RGB mean values
        #     final_sig.append(MeanRGB(maskT, frame, img_rgb, final_sig[-1], min_value, max_value))
        # else:
        #     temp, min_value, max_value = spartialAverage(maskT, frame)
        #     final_sig.append(temp)
        # end = time.time()
        # print(end - start)

        #  Those rgb values are appened to final_sig which will be used for spo2 estimation
        if cv2.waitKey(25) & 0xFF == ord('q'):  # end the loop if your press 'q' or video reaches end
            break
        # if frameCount==32:
        #     break

  # Break the loop
    else:
        break

# When everything done, release
# the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()


# For debugging you can try saving the signal as a pickle and run usingPick.py
pickle.dump(final_sig, open("spo2.p", "wb"))
print(final_sig)
# the final signal list is sent to SPooEsitmate function with length of the video
result = SPooEsitmate(final_sig, length, length, seconds)
print(result)

# TODO: Detrended into better results
# z1=[item[0] for item in final_sig]
# detrended_R=dethreding(z1)
# print(detrended_R)
#
# z2=[item[1] for item in final_sig]
# detrended_G=dethreding(z2)
# print(detrended_R)
#
# z3=[item[2] for item in final_sig]
# detrended_B=dethreding(z3)
# print(detrended_R)
