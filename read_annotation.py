#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 15:47:47 2021

@author: tarek
"""

import cv2

import numpy as np
from random import seed
from random import randint
# Blue color in BGR 
color = [(255, 0, 0),(0, 255, 0),(0, 0, 255), (255,255,255)]
  
# Line thickness of 2 px 
thickness = 2

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name


cap = cv2.VideoCapture('/home/tarek/Grenoble M2/Computer vision /AVDIAR_All/AVDIAR_All/Seq01-1P-S0M1/Video/Seq01-1P-S0M1_CAM1.mp4')
# cap = cv2.VideoCapture('/home/tarek/Grenoble M2/Computer vision /AVDIAR_All/AVDIAR_All/Seq02-1P-S0M1/Video/Seq02-1P-S0M1_CAM1.mp4')
# cap = cv2.VideoCapture('/home/tarek/Grenoble M2/Computer vision /AVDIAR_All/AVDIAR_All/Seq09-3P-S1M1/Video/Seq09-3P-S1M1_CAM1.mp4')
# cap = cv2.VideoCapture('/home/tarek/Grenoble M2/Computer vision /AVDIAR_All/AVDIAR_All/Seq27-3P-S1M1/Video/Seq27-3P-S1M1_CAM1.mp4')


f = open("/home/tarek/Grenoble M2/Computer vision /AVDIAR_All/AVDIAR_All/Seq01-1P-S0M1/GroundTruth/face_bb.txt", "r")
line = f.readline().split(',') 
 
# Check if camera opened successfully
if (cap.isOpened()== False):
  print("Error opening video stream or file")
 
nb_frame = 0
i= 51  #21   23
# Read until video is completed
while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        while line and nb_frame == int(line[0]):
          
          print(nb_frame, line)
          x =int(float(line[2]))
          y= int(float(line[3]))   
          w=int(float(line[4]))
          h=int(float(line[5]))
          
          # print (frame.shape)
          
          x1= randint(0,frame.shape[1]-x)

          y1=randint(0,frame.shape[0]-y)
        
          if nb_frame%100==0:
               img= frame[y:y+h,x:x+w]

               img1= frame[y1:y1+h,x1:x1+w]
               filename = './faces/AVDIARface{}.jpg'.format(i)
               filename1 = './back/AVDIARback{}.jpg'.format(i)
               cv2.imwrite(filename, img)
               cv2.imwrite(filename1, img1) 
               i+=1


          # cv2.rectangle(frame, (int(float(line[2])),int(float(line[3]))), (int(float(line[2]))+int(float(line[4])) ,int(float(line[3]))+int(float(line[5]))), color[int(line[1])-1], thickness)
          # (571.87, 71.999) (78.749, 80.624)
          line = f.readline()
          if line:
            line = line.split(',') 
        # Display the resulting frame
        cv2.imshow('Frame', frame)
 
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        nb_frame +=1    
    # Break the loop
    else:
        break
 
# When everything done, release the video capture object
cap.release()
print ("the final image number",i)
 
# Closes all the frames
# cv2.destroyAllWindows()