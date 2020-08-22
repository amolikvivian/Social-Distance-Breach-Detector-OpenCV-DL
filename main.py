import cv2
import time
import imutils
import argparse
import numpy as np


from math import pow, sqrt
from imutils.video import FPS
from imutils.video import VideoStream

FOCAL_LENGTH = 615
WARNING_LABEL = "Maintain Safe Distance. Move away!"

MODEL_WEIGHTS = 'models/SSD_MobileNet.caffemodel'
MODEL_CONFIG = 'models/SSD_MobileNet_prototxt.txt'

#Loading Caffe Model
print('[Status] Loading Model...')
nn = cv2.dnn.readNetFromCaffe(MODEL_CONFIG, MODEL_WEIGHTS)

#Initialize Video Stream (Use src = 0 for Webcam or src = 'path to video input')
print('[Status] Starting Video Stream...')
vs = VideoStream(src = 0).start()
time.sleep(0.1)
fps = FPS().start()

#Loop Video Stream
while True:
    
    #Resize Frame to 600 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    frame = cv2.resize(frame, (0, 0), None, 1.5, 1.5)

    #Converting Frame to Blob
    (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    #Passing Blob through network to detect and predict
    nn.setInput(blob)
    detections = nn.forward()

        
    #Creating position and coordinate sets
    pos = {}
    coordinates = {}

    #Loop over the detections
    for i in np.arange(0, detections.shape[2]):

        #Extracting the confidence of predictions
        confidence = detections[0, 0, i, 2]

        #Filtering out weak predictions
        if confidence > 0.5:

            #Extracting the index of the labels from the detection
            object_id = int(detections[0, 0, i, 1])

            #Identifying only Person as detected object
            if(object_id == 15):
                    
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype('int')

                #Draw the prediction on the frame
                label = "Person: {:.2f}%".format(confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY), (10,255,0), 2)

                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, (20,255,0), 1)

                coordinates[i] = (startX, startY, endX, endY)

                #Mid point of bounding box
                midX = abs((startX + endX) / 2)
                midY = abs((startY + endY) / 2)

                ht = round(endY-startY,4)

                #Distance from camera based on triangle similarity
                distance = (FOCAL_LENGTH * 165) / ht
                    
                #Mid-point of bounding boxes (in cm) based on triangle similarity
                midX_cm = (midX * distance) / FOCAL_LENGTH
                midY_cm = (midY * distance) / FOCAL_LENGTH
                
                #Appending the mid points of bounding box and distance between detected object and camera 
                pos[i] = (midX_cm, midY_cm, distance)
           
    
    proximity = []
    

    #Looping over positions of bounding boxes in frame
    for i in pos.keys():
        for j in pos.keys():
            if i < j:
                dist = sqrt((pos[i][0] - pos[j][0])**2 + (pos[i][1] - pos[j][1])**2 + (pos[i][2] - pos[j][2])**2)

                #Checking threshold distance - 175 cm
                if dist < 175:
                        
                    proximity.append(i)
                    proximity.append(j)
                    
                    cv2.putText(frame, WARNING_LABEL, (50,50), cv2.FONT_HERSHEY_DUPLEX, 0.5, [0,255,0], 1)
            
                        
    for i in pos.keys():
            
        if i in proximity:
            color = [0,0,255]
        else:
            color = [0,255,0]
                
        (x, y, w, h) = coordinates[i]

        cv2.rectangle(frame, (x, y), (w, h), color, 2)
                            
    cv2.imshow('Live', frame)
        
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    fps.update()

fps.stop()
print("[INFO]Elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO]Approx. FPS:  {:.2f}".format(fps.fps()))
