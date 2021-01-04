'''
    File name         : object_tracking.py
    File Description  : Multi Object Tracker Using Kalman Filter
                        and Hungarian Algorithm
    Author            : Srini Ananthakrishnan
    Date created      : 07/14/2017
    Date last modified: 07/16/2017
    Python Version    : 2.7
'''

# Import python libraries
import cv2
import copy
from detectors_Coord import Detectors
from tracker import Tracker
import numpy as np

def checkiou(iterable):
    for element in iterable:
        if element>0:
            return True
    return False


# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
	boxes = np.array(boxes)
    # if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
	# initialize the list of picked indexes	
	pick = []
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2]+boxA[0], boxB[2]+boxB[0])
    yB = min(boxA[3]+boxB[1], boxB[3]+boxB[1])
	# compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
    #iou = interArea / float(boxAArea + boxBArea - interArea)
    iou = interArea / float(boxAArea)

	# return the intersection over union value
    return iou

def main():
    """Main function for multi object tracking
    Usage:
        $ python2.7 objectTracking.py
    Pre-requisite:
        - Python2.7
        - Numpy
        - SciPy
        - Opencv 3.0 for Python
    Args:
        None
    Return:
        None
    """
    errors = 0
    total_faces = 0
    # Create opencv video capture object
    cap = cv2.VideoCapture('/home/tarek/Grenoble M2/Computer vision /AVDIAR_All/AVDIAR_All/Seq27-3P-S1M1/Video/Seq27-3P-S1M1_CAM1.mp4')
    f = open('/home/tarek/Grenoble M2/Computer vision /AVDIAR_All/AVDIAR_All/Seq27-3P-S1M1/GroundTruth/face_bb.txt', "r")
    # cap = cv2.VideoCapture('/home/tarek/Grenoble M2/Computer vision /AVDIAR_All/AVDIAR_All/Seq01-1P-S0M1/Video/Seq01-1P-S0M1_CAM1.mp4')
    line = f.readline().split(',') 

    # Create Object Detector
    detector = Detectors()

    # Create Object Tracker
    tracker = Tracker(160, 30, 5, 100)

    # Variables initialization
    track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                    (0, 255, 255), (255, 0, 255), (255, 127, 255),
                    (127, 0, 255), (127, 0, 127)]
    pause = False
    nb_frame = 0 #########
    
    
    # Infinite loop to process video frames
    while(cap.isOpened()):
          # Capture frame-by-frame
          coord_GT = [] #ground_truth coordinates
          ret, frame = cap.read()
          print("error",errors)

          
#          print(nb_frame)
          if ret == True:
              # print(nb_frame)
              while line and nb_frame == int(line[0]): #finchè il numero della linea 0 rimane il frame corrente
                  coord_GT.append((int(float(line[2])),int(float(line[3])), int(float(line[4]))
                                   ,int(float(line[5]))))
                  cv2.rectangle(frame, (int(float(line[2])),int(float(line[3]))),
                                (int(float(line[2]))+int(float(line[4])),
                                 int(float(line[3]))+int(float(line[5]))),
                                color=(255,255,255), thickness = 2)
                  line = f.readline() #for the next cycle
                  if line:
                    line = line.split(',') 
                    
              nb_frame +=1

              # Detect and return centeroids of the objects in the frame
              centers, coord = detector.Detect(frame)
              coord = list(non_max_suppression_fast(coord, 0.3))
              for i in range(len(coord)):
                  tmp_coord = coord[i]
                  # cv2.rectangle(frame, (tmp_coord[0], tmp_coord[1]), 
                  #               (tmp_coord[0]+tmp_coord[2],tmp_coord[1]+tmp_coord[3]),
                  #               color = (0, 255, 0), thickness=2)
                  
                  

              # If centroids are detected then track them

              if (len(centers) > 0):
                    # Track object using Kalman Filter
                    tracker.Update(centers)
                    # For identified object tracks draw tracking line
                    # Use various colors to indicate different track_id
                    for i in range(len(tracker.tracks)):
                        if (len(tracker.tracks[i].trace) > 1):
                            try:
                                for j in range(len(tracker.tracks[i].trace)-1):
                                    # Draw trace line
                                    x1 = tracker.tracks[i].trace[j][0][0]
                                    y1 = tracker.tracks[i].trace[j][1][0]
                                    x2 = tracker.tracks[i].trace[j+1][0][0]
                                    y2 = tracker.tracks[i].trace[j+1][1][0]
                                    clr = tracker.tracks[i].track_id % 9
                                    # cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                    #          track_colors[clr], 2)
                                    cv2.rectangle(frame,(int(x2)-int(coord[j][2]/2), int(y2)-int(coord[j][3]/2)),
                                                  (int(x2)+int(coord[j][2]/2), int(y2)+int(coord[j][3]/2)),
                                                  color = (0, 255, 100), thickness=2)
                                    
                            except :
                                pass
              
              
              total_faces += len(coord_GT)
    #          print('detected coordinates')
    #          print(coord)
    #          print('ground_truth coordinates')
    #          print(coord_GT)
    #          print('##################################')
              if len(coord) != 0:
                  for i in range(len(coord)):
                      iou = []
                      for j in range(len(coord_GT)):
                          iou.append(bb_intersection_over_union(coord[i], coord_GT[j]))
    #                      print(iou)
                      if checkiou(iou) == False:
    #                      value_greater_than_thresh = [k for k in iou if k>0]
    #                      print(value_greater_than_thresh)
    #                      if len(value_greater_than_thresh) == 0:
                          errors += 1
                              
              else:
                  errors += len(coord_GT)
              
                
              
    
              
         
                  # Display the resulting tracking frame
              cv2.imshow('Tracking', frame)
               # nb_frame +=1
            # Press Q on keyboard to  exit
              if cv2.waitKey(10) & 0xFF == ord('q'):
                  break
          
    


    cap.release()
    cv2.destroyAllWindows()
    print(errors)
    print(total_faces)


if __name__ == "__main__":
    # execute main
    main()
