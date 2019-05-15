import numpy as np
import argparse
import time
import cv2
import os
import face_recognition
import pickle
from utils import BD
from roma_bd import BD_roman
import matplotlib.pyplot as plt
import glob
from scipy.ndimage import rotate

import dlib
tracker = dlib.correlation_tracker()
tracking_face = 0

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe("./PersonDetectors/MobileNetSSD_deploy.prototxt.txt", "./PersonDetectors/MobileNetSSD_deploy.caffemodel")

confidence_thresh = 0.3
threshold = 0.3

video_capture = cv2.VideoCapture('test_out_04.avi')


# samples = glob.glob('foto/*')
# known_face_names = []
# known_face_encodings = []
# for image in samples:
#     known_face_names.append(image.split('/')[1].split('.')[0])
    
#     face = face_recognition.load_image_file(image)
#     face = rotate(face, -90)
    
#     known_face_encodings.append(face_recognition.face_encodings(face)[0])

with open('data_roma.pickle', 'rb') as f:
     data = pickle.load(f)

known_face_names, known_face_encodings = data.get_data()


face_locations = []
face_encodings = []
face_names = []
process_this_frame = True


labelsPath = "./coco.names"
LABELS = open(labelsPath).read().strip().split("\n")
 
# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")


while True:
    ret, frame = video_capture.read()
    image_full = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
    rgb_small_frame = image_full[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_small_frame)
    if process_this_frame:

        gray = cv2.cvtColor(image_full, cv2.COLOR_BGR2GRAY)

        # face_locations = face_recognition.face_locations(image_full)
        face_locations = faceCascade.detectMultiScale(gray, 1.3, 5)

        face_locations = [(_y, _x+_w, _y+_h, _x) for (_x,_y,_w,_h) in face_locations]
        # face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:

            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # раскоментить в продакшн
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # top *= 2
        # left 
        # top 
        # right +=x
        # bottom +=y
        # right *= 2
        # bottom *= 2
        # left *= 2

        cv2.rectangle(image_full, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(image_full, (left, bottom - 10), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image_full, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    #pos detection

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

                

                
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(image_full, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                cv2.putText(image_full, 'student', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)
    cv2.imshow("Image", image_full)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()