import numpy as np
import argparse
import time
import cv2
import os
import face_recognition

# net = cv2.dnn.readNetFromDarknet("./yolov3.cfg", "./yolov3.weights")
net = cv2.dnn.readNetFromDarknet("./yolov3-tiny.cfg", "./yolov3-tiny.weights")
video_capture = cv2.VideoCapture('./../test_out_03.avi')
# video_capture = cv2.VideoCapture(1)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

CLASSESPath = "./coco.names"
CLASSES = open(CLASSESPath).read().strip().split("\n")
 
# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(CLASSES), 3),
	dtype="uint8")

confidence_thresh = 0.3
threshold = 0.3
while True:
    ret, frame = video_capture.read()
    image_full = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
    image_yolo = cv2.resize(image_full, (0, 0), fx=1, fy=1)
    (H, W) = image_yolo.shape[:2]
    blob = cv2.dnn.blobFromImage(image_yolo, 1 / 255.0, (250, 250), #416 416
        swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)

    boxes = []
    confidences = []
    classIDs = []
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
    
            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > confidence_thresh:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                
                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
    
                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_thresh, threshold)
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            if CLASSES[classIDs[i]]== "person":
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                # x /= 0.35
                # y /= 0.35
                # w /= 0.35
                # h /= 0.35
                x = int(x)
                y = int(y)
                w = int(w)
                h = int(h)
                crop = image_full[y:y+h, x:x+w]
                crop = crop[:, :, ::-1]
                face_locations = face_recognition.face_locations(crop)
                for (top, right, bottom, left) in face_locations:
                    cv2.rectangle(image_full, (x+left, y+top), (x+right, y+bottom), (0, 0, 255), 2)
                # draw a bounding box rectangle and label on the image
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(image_full, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(CLASSES[classIDs[i]], confidences[i])
                cv2.putText(image_full, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)
    cv2.imshow("Image", image_full)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()