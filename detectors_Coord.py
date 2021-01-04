'''
    File name         : detectors.py
    File Description  : Detect objects in video frame
    Author            : Srini Ananthakrishnan
    Date created      : 07/14/2017
    Date last modified: 07/16/2017
    Python Version    : 2.7
'''

# Import python libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt


from mtcnn.mtcnn import MTCNN

#%%

# set to 1 for pipeline images
debug = 0


class Detectors(object):
    """Detectors class to detect objects in video frame
    Attributes:
        None
    """
    def __init__(self):
        """Initialize variables used by Detectors class
        Args:
            None
        Return:
            None
        """
        self.detector= MTCNN()


    def Detect(self, frame):
        """Detect objects in video frame using following pipeline
            - Convert captured frame from BGR to GRAY
            - Perform Background Subtraction
            - Detect edges using Canny Edge Detection
              http://docs.opencv.org/trunk/da/d22/tutorial_py_canny.html
            - Retain only edges within the threshold
            - Find contours
            - Find centroids for each valid contours
        Args:
            frame: single video frame
        Return:
            centers: vector of object centroids in a frame
        """
        coord = []
        centers = []  # vector of object centroids in a frame       

        faces = self.detector.detect_faces(frame)
        
        for result in faces:
	    	# get coordinates
            x,y,width,height = result['box']
            coord.append((x,y,width,height))
            #cv2.rectangle(frame, (x, y), (x+width,y+height), (0, 255, 0), 2)
            # Closes all the frames
            centerX = int(x + 0.5 * (width - 1));
            centerY = int(y + 0.5 * (height - 1));
            b = np.array([[centerX], [centerY]])
            centers.append(b)

        #cv2.imshow("Window", frame)

     
        return centers, coord
    
   