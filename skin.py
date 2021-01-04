
import numpy as np
import cv2
from sklearn.cluster import KMeans
from collections import Counter
import imutils
import pprint
from matplotlib import pyplot as plt


def extractSkin(image):
    # Taking a copy of the image
    img = image.copy()
    # Converting from BGR Colours Space to HSV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Defining HSV Threadholds
    lower_threshold = np.array([0, 48, 80], dtype=np.uint8)
    upper_threshold = np.array([20, 255, 255], dtype=np.uint8)

    # Single Channel mask,denoting presence of colours in the about threshold
    skinMask = cv2.inRange(img, lower_threshold, upper_threshold)

    # Cleaning up mask using Gaussian Filter
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)

    # Extracting skin from the threshold mask
    skin = cv2.bitwise_and(img, img, mask=skinMask)

    # Return the Skin image
    return cv2.cvtColor(skin, cv2.COLOR_HSV2BGR)


def removeBlack(estimator_labels, estimator_cluster):

    # Check for black
    hasBlack = False

    # Get the total number of occurance for each color
    occurance_counter = Counter(estimator_labels)

    # Quick lambda function to compare to lists
    def compare(x, y): return Counter(x) == Counter(y)

    # Loop through the most common occuring color
    for x in occurance_counter.most_common(len(estimator_cluster)):

        # Quick List comprehension to convert each of RBG Numbers to int
        color = [int(i) for i in estimator_cluster[x[0]].tolist()]

        # Check if the color is [0,0,0] that if it is black
        if compare(color, [0, 0, 0]) == True:
            # delete the occurance
            del occurance_counter[x[0]]
            # remove the cluster
            hasBlack = True
            estimator_cluster = np.delete(estimator_cluster, x[0], 0)
            break

    return (occurance_counter, estimator_cluster, hasBlack)


def getColorInformation(estimator_labels, estimator_cluster, hasThresholding=False):

    # Variable to keep count of the occurance of each color predicted
    occurance_counter = None

    # Output list variable to return
    colorInformation = []

    # Check for Black
    hasBlack = False

    # If a mask has be applied, remove th black
    if hasThresholding == True:

        (occurance, cluster, black) = removeBlack(
            estimator_labels, estimator_cluster)
        occurance_counter = occurance
        estimator_cluster = cluster
        hasBlack = black

    else:
        occurance_counter = Counter(estimator_labels)

    # Get the total sum of all the predicted occurances
    totalOccurance = sum(occurance_counter.values())

    # Loop through all the predicted colors
    for x in occurance_counter.most_common(len(estimator_cluster)):

        index = (int(x[0]))

        # Quick fix for index out of bound when there is no threshold
        index = (index-1) if ((hasThresholding & hasBlack)
                              & (int(index) != 0)) else index

        # Get the color number into a list
        color = estimator_cluster[index].tolist()

        # Get the percentage of each color
        color_percentage = (x[1]/totalOccurance)

        # make the dictionay of the information
        colorInfo = {"cluster_index": index, "color": color,
                     "color_percentage": color_percentage}

        # Add the dictionary to the list
        colorInformation.append(colorInfo)

    return colorInformation


def extractDominantColor(image, number_of_colors=5, hasThresholding=False):

    # Quick Fix Increase cluster counter to neglect the black(Read Article)
    if hasThresholding == True:
        number_of_colors += 1

    # Taking Copy of the image
    img = image.copy()

    # Convert Image into RGB Colours Space
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Reshape Image
    img = img.reshape((img.shape[0]*img.shape[1]), 3)

    # Initiate KMeans Object
    estimator = KMeans(n_clusters=number_of_colors, random_state=0)

    # Fit the image
    estimator.fit(img)

    # Get Colour Information
    colorInformation = getColorInformation(
        estimator.labels_, estimator.cluster_centers_, hasThresholding)
    return colorInformation


def plotColorBar(colorInformation):
    # Create a 500x100 black image
    color_bar = np.zeros((100, 500, 3), dtype="uint8")

    top_x = 0
    for x in colorInformation:
        bottom_x = top_x + (x["color_percentage"] * color_bar.shape[1])

        color = tuple(map(int, (x['color'])))

        cv2.rectangle(color_bar, (int(top_x), 0),
                      (int(bottom_x), color_bar.shape[0]), color, -1)
        top_x = bottom_x
    return color_bar


"""## Section Two.4.2 : Putting it All together: Pretty Print
The function makes print out the color information in a readable manner
"""


def prety_print_data(color_info):
    for x in color_info:
        print(pprint.pformat(x))
        print()
#%%

alpha = 1
# Blue color in BGR 
#color = [(255, 0, 0),(0, 255, 0),(0, 0, 255), (255,255,255)]
i = 0  
# Line thickness of 2 px 
thickness = 2

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('/home/tarek/Grenoble M2/Computer vision /AVDIAR_All/AVDIAR_All/Seq01-1P-S0M1/Video/Seq01-1P-S0M1_CAM1.mp4')

# cap = cv2.VideoCapture('/home/tarek/Grenoble M2/Computer vision /AVDIAR_All/AVDIAR_All/Seq27-3P-S1M1/Video/Seq27-3P-S1M1_CAM1.mp4')

#cap = cv2.VideoCapture('./AVDIAR_All/Seq03-1P-S0M1/Video/Seq03-1P-S0M1_CAM1.mp4')

#cap = cv2.VideoCapture('./AVDIAR_All/Seq27-3P-S1M1/Video/Seq27-3P-S1M1_CAM1.mp4')
#cap = cv2.VideoCapture('./AVDIAR_All/Seq43-2P-S0M0/Video/Seq43-2P-S0M0_CAM1.mp4')

f = open("/home/tarek/Grenoble M2/Computer vision /AVDIAR_All/AVDIAR_All/Seq01-1P-S0M1/GroundTruth/face_bb.txt", "r")
#f = open("./AVDIAR_All/Seq27-3P-S1M1/GroundTruth/face_bb.txt", "r")

line = f.readline().split(',') 
 
# Check if camera opened successfully
if (cap.isOpened()== False):
    print("Error opening video stream or file")
frame_x = 0
nb_frame = 0
ex_frame = np.zeros((450, 720,3))
# Constants for finding range of skin color in YCrCb
#min_YCrCb = np.array([0,133,77],np.uint8)
#max_YCrCb = np.array([255,173,127],np.uint8)
# Read until video is completed
while (cap.isOpened()):
    # Capture frame-by-frame
    frame_x += 1 
    ret,frame = cap.read()
    #if ret:
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
        
        #print(frame.shape)
#        for i in range(2):
#            frame[:,:,i] = cv2.equalizeHist(frame[:,:,i])
        #frame[:,:,0] = np.zeros((450,720))
       
        #frame = frame[:,:,2] #cr channel


    if ret == True:
        while line and nb_frame == int(line[0]):
          
          
          print(nb_frame)#, line)
          
          
          
          
          line = f.readline()
          
          if line:
            line = line.split(',') 
        
          
          
          skin = extractSkin(frame) 

#            
          cv2.imshow('Frame', skin)
            
        
        
        # Press Q on keyboard to  exit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        nb_frame +=1    
    # Break the loop
    else:
        break
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()
