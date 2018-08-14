

import cv2
import numpy as np

# Create our body classifier
body_classifier = cv2.CascadeClassifier('pedestrian_tracking.xml')

# video capture for video file
cap = cv2.VideoCapture(0)

# Loop once video is successfully loaded
while True:
    
    # Read first frame
    ret, frame = cap.read()
    frame = cv2.resize(frame, None,fx=0.4, fy=0.5, interpolation = cv2.INTER_LINEAR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Pass frame to our body classifier
    bodies = body_classifier.detectMultiScale(gray, 1.3, 3)
    
    # to draw a rectangle around detected object
    for (x,y,w,h) in bodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #cv2.imshow('Pedestrians', frame)

    #if cv2.waitKey(1) == 13: #13 is the Enter Key
     #   break
    cv2.imshow('img',frame)
    k = cv2.waitKey(30) & 0xff
    if k == 13:
        break

cap.release()
cv2.destroyAllWindows()
