import numpy as np
import argparse
import time
import cv2
import os
import face_recognition

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe("./MobileNetSSD_deploy.prototxt.txt", "./MobileNetSSD_deploy.caffemodel")

confidence_thresh = 0.1
threshold = 0.1
# video_capture = cv2.VideoCapture('./../test_out_04.avi')
video_capture = cv2.VideoCapture(1)
while True:
    ret, frame = video_capture.read()
    image_full = cv2.resize(frame, (0, 0), fx=0.7, fy=0.7)
    image_yolo = cv2.resize(image_full, (0, 0), fx=1, fy=1)
    (H, W) = image_yolo.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image_yolo, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    boxes = []
    confidences = []
    classIDs = []
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]
        confidences.append(confidence)
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > confidence_thresh:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            w = box[2] - box[0]
            h = box[3] - box[1]
            box[2] = w
            box[3] = h
            boxes.append(box)
            classIDs.append(idx)

            # (startX, startY, endX, endY) = box.astype("int")
    # idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_thresh, threshold)
    if len(classIDs) > 0:
        # loop over the indexes we are keeping
        for i in range(len(classIDs)):
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