#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 21:42:31 2021

@author: tarek
"""
# example of inference with a pre-trained coco model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from matplotlib import pyplot
from matplotlib.patches import Rectangle

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# face detection with mtcnn on a photograph
class TestConfig(Config):
     NAME = "test"
     GPU_COUNT = 1
     IMAGES_PER_GPU = 1
     NUM_CLASSES = 1 + 80


# define the model
rcnn = MaskRCNN(mode='inference', model_dir='./', config=TestConfig())
# load coco model weights
rcnn.load_weights('mask_rcnn_coco.h5', by_name=True)


#%%

cap = cv2.VideoCapture('/home/tarek/Grenoble M2/Computer vision /AVDIAR_All/AVDIAR_All/Seq27-3P-S1M1/Video/Seq27-3P-S1M1_CAM1.mp4')

i=0

if (cap.isOpened()== False):
    print("Error opening video stream or file")
    
ret,frame = cap.read()

if i==0:
    
    test_img = frame
    # test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

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

def draw_image_with_boxes(filename, boxes_list):
     # load the image
     data = test_img
     # plot the image
     pyplot.imshow(data)
     # get the context for drawing boxes
     ax = pyplot.gca()
     # plot each box
     for box in boxes_list:
          # get coordinates
          y1, x1, y2, x2 = box
          # calculate width and height of the box
          width, height = x2 - x1, y2 - y1
          # create the shape
          rect = Rectangle((x1, y1), width, height, fill=False, color='red')
          # draw the box
          ax.add_patch(rect)
     # show the plot
     pyplot.show()
     
     
results = rcnn.detect([test_img], verbose=0)


draw_image_with_boxes(test_img, results[0]['rois'])




#%%

# face detection with mtcnn on a photograph
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mtcnn.mtcnn import MTCNN

# draw an image with detected objects
def draw_image_with_boxes(filename, result_list):
	# load the image
	data = filename
	# plot the image
	pyplot.imshow(data)
	# get the context for drawing boxes
	ax = pyplot.gca()
	# plot each box
	for result in result_list:
		# get coordinates
		x, y, width, height = result['box']
		# create the shape
		rect = Rectangle((x, y), width, height, fill=False, color='red')
		# draw the box
		ax.add_patch(rect)
	# show the plot
	pyplot.show()

filename = test_img
# load image from file
# create the detector, using default weights
detector = MTCNN()
# detect faces in the image
faces = detector.detect_faces(test_img)
# display faces on the original image
draw_image_with_boxes(filename, faces)