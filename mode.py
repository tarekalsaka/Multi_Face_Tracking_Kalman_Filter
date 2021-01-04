#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 14:41:00 2021

@author: tarek
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


cap = cv2.VideoCapture('/home/tarek/Grenoble M2/Computer vision /AVDIAR_All/AVDIAR_All/Seq27-3P-S1M1/Video/Seq27-3P-S1M1_CAM1.mp4')

debug=1
i=0
#%%



#%%
i=0

if (cap.isOpened()== False):
    print("Error opening video stream or file")
    
ret,frame = cap.read()

if i==0:
    
    test_img = frame
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    plt.figure()
    plt.imshow(test_img)
    i +=1
    
imgFile = 'img_532.jpg'

test2_image= cv2.imread(imgFile)
test2_image = cv2.cvtColor(test2_image, cv2.COLOR_BGR2GRAY)

plt.figure()
plt.imshow(test2_image)

cap.release()


#%%


#importing the model trained in lab2
from tensorflow.keras.models import load_model

model = load_model('convNetGreyscal.h5')

model.summary()

#%%
# Window_size = [20,24,29,35,42,50,60,70,99,120,140,170,200]

Window_size=[30]
def checkscore(iterable):
    for element in iterable:
        if element>0.8:
            return True
    return False

def alg_pyramid(image, minSize=(16, 16)):
    pyramid = []
    sigma = 1
    #first level
    image = cv2.GaussianBlur(image, (7,7), sigma)

    pyramid.append(image)
    while True: #while  removing n_level
        sigma=1
        image = cv2.GaussianBlur(image, (7,7), sigma)
        sigma = sigma*np.sqrt(2)
        image = cv2.GaussianBlur(image, (9,9), sigma)
        # w = int(image.shape[1] /2)
        # h = int(image.shape[0] /2)
#         image = imutils.resize(image, width=w, height=h)
        #image = cv2.resize(image, (image.shape[0]/2, image.shape[1]/2), interpolation= cv2.INTER_NEAREST)
        image = image[::2,::2] #downsampling S_2
        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        pyramid.append(image)

            
    return pyramid


def sliding_window2(image, stepSize, windowSize):
    
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            
            yield (x, y, image[y:y + int(windowSize), x:x + int(windowSize)])
            # print("x",x,"y",y,"windowSize",windowSize)
            if x+int(windowSize)>image.shape[1]:
                break
        if y+int(windowSize)>image.shape[0] :
            break
        
        
def Detect_Face(image):
    
    temp_img_pyr=alg_pyramid(image)
    score=[]

    
    for index1,win in enumerate( Window_size):
        tem_savewindow_coord=[]
        savewindow=[]
        temp=[]
        
        for i in range (len(temp_img_pyr)):
            print("level",i)
            
            
            # clo = cv2.cvtColor(temp_img_pyr[0], cv2.COLOR_RGB2BGR)
            clo=temp_img_pyr[0]
            
        
    
            try:
                for (x, y, window) in sliding_window2(temp_img_pyr[0], int(int(win)/2), int(win)):
                    
                                
            
                    #reshaping the window for go in the network..
                    #(l,t,r,b)
                    window_coord = x,y,x + window.shape[1], y + window.shape[0]
                    
            
                    window= window/255
                    window = cv2.resize(window, (32,32))
                    window = window.reshape(window.shape[0], window.shape[1],1)
                    savewindow.append(window)
                    tem_savewindow_coord.append(window_coord)
                img_array =np.array(savewindow)
                #             print(img_array222.shape)
                y_pred_prob = model.predict(img_array)
                # print (type(y_pred_prob))
                
                y_pred = np.argmax(y_pred_prob, axis=1)
                print (y_pred)
                
                for idx, pre in np.ndenumerate(y_pred):
                    
                    if pre ==1:# face
                        # score.append(y_pred_prob[idx[0]])
                    
                        if checkscore(y_pred_prob[idx]):
                            print (y_pred_prob[idx])

                            cv2.rectangle(clo, (tem_savewindow_coord[idx[0]][0], tem_savewindow_coord[idx[0]][1]), (tem_savewindow_coord[idx[0]][2],tem_savewindow_coord[idx[0]][3]), (0, 255, 0), 2)
                            cv2.imshow("Window", clo)
                            if cv2.waitKey(1000) & 0xFF == ord('q'):
                                break
     
                            # Closes all the frames
                            cv2.destroyAllWindows()
                        
            
            except:
                
                print ("no index")
                
                
    return score



value = Detect_Face(test_img)
                
                
            
  #%%


if (cap.isOpened()== False):
    print("Error opening video stream or file")
while (cap.isOpened()):    # read frame-by-frame
    ret, frame = cap.read()

    # set the frame to gray as we do not need color, save up the resources
    if ret:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if i==0:
    
        test_img = gray_frame
      
        plt.figure()
        plt.imshow(test_img)
        i +=1
    
    # pass the frame to the classifier
    persons_detected = classifier.detectMultiScale(gray_frame, 1.3, 5)

    # check if people were detected on the frame
    # try:
    #     human_count = persons_detected.shape[0]
    # except :
    #     human_count = 0
    
    # # extract boxes so we can visualize things better
    # # for actual deployment with hardware, not needed
    # for (x, y, w, h) in persons_detected:
    #     cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)    
    # cv2.imshow('Video footage', frame)

    # if (cv2.waitKey(1) & 0xFF == ord('q')):
    #     break
    
cap.release()
          