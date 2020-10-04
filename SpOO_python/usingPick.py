import cv2
import imutils
import numpy as np
import time
from face_detection import FaceDetection
from scipy import signal
import sys
from numpy.linalg import inv


frameCount=1
final_sig=[]
global gMask
global gBox
import pickle

np.set_printoptions(threshold=sys.maxsize)


def dethreding(z):

    return  signal.detrend(z)

    # T=len(z)
    # lamba=10
    # I  = np.identity(T)
    # filt = [1,-2,1]* np.ones((1,T-2),dtype=np.int).T
    # D2 = scipy.sparse.spdiags(data.T, (range(0,3)),T-2,T)
    # detrended_sig =
    # return z_stat

def face_detect_and_thresh(frame):
    lower = np.array([0, 1,0], dtype = "uint8")
    upper = np.array([179, 255, 255], dtype = "uint8")
    fd = FaceDetection()
    frame, face_frame, ROI1, ROI2, status, mask,extra,box = fd.face_detect(frame)
    frame=extra
    # cv2.imshow("face0",extra)
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)
    # apply a series of erosions and dilations to the mask
    # using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
    # blur the mask to help remove noise, then apply the
    # mask to the frame
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    skin = cv2.bitwise_and(frame, frame, mask = skinMask)
    gMask=skinMask
    global gBox
    gBox = box
    # show the skin in the image along with the mask
    # cv2.imshow("images", np.hstack([frame, skin]))
    # cv2.imshow("mask",np.hstack([mask]))


    for _ in range(np.shape(mask)[0]):
        for j in range(np.shape(mask)[1]):
            if mask[_][j]==1:
                mask[_][j]=1
            else :
                mask[_][j]=0
    mask=mask.astype(int)
    return skin,frame,mask

def spartialAverage(thresh,frame):
    a=list(np.argwhere(maskT==1))
    # x=[i[0] for i in a]
    # y=[i[1] for i in a]
    # p=[x,y]
    ind_img=(np.vstack((a)))
    sig_fin=np.zeros([np.shape(ind_img)[0],3])
    test_fin=[]
    for i in range(np.shape(ind_img)[0]):
        sig_temp=frame[ind_img[i,0],ind_img[i,1],:]
        sig_temp = sig_temp.reshape((1, 3))
        if sig_temp.any()!=0 :
            sig_fin=np.concatenate((sig_fin,sig_temp))
    print(sig_fin.shape)
    for _ in sig_fin:
        if sum(_)>0:
            # test_fin=np.concatenate((test_fin,_))
            test_fin.append(_)
    img_rgb_mean=np.nanmean(test_fin,axis=0)
    return (img_rgb_mean)



def face_detect_and_thresh2(frame):
    lower = np.array([0, 1,0], dtype = "uint8")
    upper = np.array([179, 255, 255], dtype = "uint8")
    face_frame=frame[gBox[0]:gBox[1],gBox[2]:gBox[3]]
    # cv2.imshow("face_testing",face_frame)
    frame=face_frame
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)
    # apply a series of erosions and dilations to the mask
    # using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
    # blur the mask to help remove noise, then apply the
    # mask to the frame
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    skin = cv2.bitwise_and(frame, frame, mask = skinMask)
    # cv2.imshow("face_testing",skin)
    return skin,frame


def preprocess(z1,z2,detrended_RGB,window_size,size_video,duration,frame):
    temp=(int(size_video/duration))
    f=347

    main_R=[]
    main_B=[]
    out=[]
    for i in range(len(detrended_RGB)-f+1):

        temp_R=z1[i:i+f-1]
        temp_B=z2[i:i+f-1]
        p=[list(a) for a in zip(temp_R, temp_B)]

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
    print(out[0])
    return out[0]


def SPooEsitmate(final_sig,video_size,frames):
    A=101.6
    B=5.834
    ten=10
    z1=[item[0] for item in final_sig]
    z3=[item[2] for item in final_sig]
    SPO_pre=[]
    for _ in range(len(z1)):
        SPO_pre.append([z1[_],z3[_]])
    Spo2 = preprocess(z1,z3,SPO_pre,ten,video_size,11,frames)

    R_temp = [item[0] for item in Spo2]
    DC_R_comp=np.mean(R_temp)
    AC_R_comp=np.std(R_temp)

    I_r=AC_R_comp/DC_R_comp

    B_temp = [item[1] for item in Spo2]
    DC_B_comp=np.mean(B_temp)
    AC_B_comp=np.std(B_temp)

    I_b=AC_B_comp/DC_B_comp
    print(I_r)
    print(I_b)
    SpO2_value=np.floor(A-B*((I_b*650)/(I_r*950)))
    return SpO2_value
# cap=cv2.VideoCapture("test.avi")
#
# if (cap.isOpened()==False):
#     print("Error opening video file")
#
# while(cap.isOpened()):
#     ret,frame = cap.read()
#     if frameCount==1:
#         firstFrame=frame
#         thresh,img_rgb,maskT=face_detect_and_thresh(firstFrame)
#         frameCount+=1
#
#     if ret == True:
#         frameCount+=1
#         print(frameCount)
#         cv2.imshow('Frame',frame)
#         thresh,img_rgb=face_detect_and_thresh2(frame)
#         final_sig.append(spartialAverage(maskT,frame))
#         if cv2.waitKey(25) & 0xFF == ord('q'):
#             break
#   # Break the loop
#     else:
#         break
final_sig = pickle.load( open( "spo2.p", "rb" ) )
print(final_sig)
video_size=121
# pickle.dump(final_sig, open( "filename.p", "wb" ) )

z1=[item[0] for item in final_sig]
detrended_R=dethreding(z1)

# print(detrended_R)
z2=[item[1] for item in final_sig]
detrended_G=dethreding(z2)

# print(detrended_G)
z3=[item[2] for item in final_sig]
detrended_B=dethreding(z3)
# print(detrended_B)

result=SPooEsitmate(final_sig,video_size,video_size)

print(94.535)
