# Social-Distance-Breach-Detector-OpenCV-DL
OpenCV with Python based project using Caffe Deep Learning Framework 
and Triangle Similarity Theorem to identify Social Distance Breaches.

# Model
Caffe Deep Learning Framework (Single Shot Detection) to identify people in frame - SSD_MobileNet.caffemodel

# Distance and Depth
Triangle Similarity Theorem to measure distance from objects to camera. Further, centroids of each bounding box are taken as reference
to find distance between two objects.

# Streaming
Local host live streaming using Flask - Python Web Framework

# Steps to Execute Code

1. Launch terminal 

2. cd to Project Directory 

3. Write the following commands in the terminal - 
  'set FLASK_APP = Main.py' -> 
  'flask run --host = 0.0.0.0'

4. Go on to preferred web browser -> 'localhost: port number being shown on terminal window'

5. To exit from stream and terminate operations -> ctrl + c

# References

1. https://www.pyimagesearch.com/2015/01/19/find-distance-camera-objectmarker-using-python-opencv/
2. https://www.pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/
