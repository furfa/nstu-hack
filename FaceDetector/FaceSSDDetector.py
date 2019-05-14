import numpy as np
import cv2
class FaceSSDDetector:
    def __init__(self, prototext, model):
        print("loading")
        self.net = cv2.dnn.readNetFromCaffe(prototext, model)
        print("loadede")
    def face_locations(self, image, conf=0.5):
        (h, w) = image.shape[:2]
        
        # cv2.imshow("croped", image)
        print(h, w)
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
	        (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        # print()
        boxes = []
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]
        
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > conf:
                # compute the (x, y)-coordinates of the bounding box for the
                # object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                
                (startX, startY, endX, endY) = box.astype("int")
                boxes.append((startY, endX, endY, startX))
        return boxes