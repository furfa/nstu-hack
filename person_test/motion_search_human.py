import numpy as np
import argparse
import time
import cv2
import os
import face_recognition
import imutils
firstFrame = None
# video_capture = cv2.VideoCapture(1)
video_capture = cv2.VideoCapture('./../test_out_04.avi')
while True:
    ret, frame = video_capture.read()
    image_full = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
    if frame is None:
        break
 
    # resize the frame, convert it to grayscale, and blur it
    # frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(image_full, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    if firstFrame is None:
        firstFrame = gray
        continue
    frameDelta = cv2.absdiff(firstFrame, gray)
    
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
 
    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    cv2.imshow("thresh", thresh)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < 500:
            continue
 
        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(image_full, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    # 	cv2.CHAIN_APPROX_SIMPLE)
    # cnts = imutils.grab_contours(cnts)
    # if the first frame is None, initialize it
    
    # if len(idxs) > 0:
    #     # loop over the indexes we are keeping
    #     for i in idxs.flatten():
    #         # extract the bounding box coordinates
    #         (x, y) = (boxes[i][0], boxes[i][1])
    #         (w, h) = (boxes[i][2], boxes[i][3])
    #         x /= 0.35
    #         y /= 0.35
    #         w /= 0.35
    #         h /= 0.35
    #         x = int(x)
    #         y = int(y)
    #         w = int(w)
    #         h = int(h)
    #         crop = image_full[y:y+h, x:x+w]
    #         crop = crop[:, :, ::-1]
    #         face_locations = face_recognition.face_locations(crop)
    #         for (top, right, bottom, left) in face_locations:
    #             cv2.rectangle(image_full, (x+left, y+top), (x+right, y+bottom), (0, 0, 255), 2)
    #         # draw a bounding box rectangle and label on the image
    #         color = [int(c) for c in COLORS[classIDs[i]]]
    #         cv2.rectangle(image_full, (x, y), (x + w, y + h), color, 2)
    #         text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
    #         cv2.putText(image_full, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
    #             0.5, color, 2)
    firstFrame = gray
    cv2.imshow("Image", image_full)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()