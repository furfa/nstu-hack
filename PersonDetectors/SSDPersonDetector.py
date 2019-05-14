import numpy as np
import time
import cv2
import os


class SSDPersonDetetor:
    def __init__(self, prototxt, caffe):
        self.net = cv2.dnn.readNetFromCaffe(prototxt,caffe)
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"]

    def person_locations(self, image, conf=0.1, thresh=0.1):
        (H, W) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()

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
            idx = int(detections[0, 0, i, 1])
            if confidence > conf and self.CLASSES[idx] == "person":
                # extract the index of the class label from the `detections`,
                # then compute the (x, y)-coordinates of the bounding box for
                # the object
                
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                w = box[2] - box[0]
                h = box[3] - box[1]
                box[2] = w
                box[3] = h
                boxes.append(box)
                # classIDs.append(idx)
        return boxes
