import numpy as np
import argparse
import time
import cv2
import os
import face_recognition
import pickle


net = cv2.dnn.readNetFromDarknet("./yolov3-tiny.cfg", "./yolov3-tiny.weights")
video_capture = cv2.VideoCapture('test_out_04.avi')
# video_capture = cv2.VideoCapture(1)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

potemin_image = face_recognition.load_image_file("ya.JPG")
potemin_face_encoding = face_recognition.face_encodings(potemin_image)[0]

vasya_image = face_recognition.load_image_file('vasya.jpg')
vasya_face_encoding = face_recognition.face_encodings(vasya_image)[0]


with open('data_v1.pickle', 'rb') as f:
     data = pickle.load(f)

known_face_names, known_face_encodings = data.get_data()



# known_face_encodings = [
#     potemin_face_encoding,
#     vasya_face_encoding
# ]
# known_face_names = [
#     'Roman',
#     'Vasya'
# ]

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
    image_yolo = cv2.resize(image_full, (0, 0), fx=0.35, fy=0.35)
    (H, W) = image_yolo.shape[:2]
    blob = cv2.dnn.blobFromImage(image_yolo, 1 / 255.0, (416, 416),
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
            if confidence > 0.3:
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
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.5)
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            x /= 0.35
            y /= 0.35
            w /= 0.35
            h /= 0.35
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)

            rgb_small_frame = image_full[y:y+h, x:x+w]
            rgb_small_frame = rgb_small_frame[:, :, ::-1]
            face_locations = face_recognition.face_locations(rgb_small_frame)
            if process_this_frame:

                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                face_names = []
                for face_encoding in face_encodings:

                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"

                    if True in matches:
                        first_match_index = matches.index(True)
                        name = known_face_names[first_match_index]

                    face_names.append(name)

            process_this_frame = not process_this_frame


    # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # top *= 2
                left +=x
                top += y
                right +=x
                bottom +=y
                # right *= 2
                # bottom *= 2
                # left *= 2

                cv2.rectangle(image_full, (left, top), (right, bottom), (0, 0, 255), 2)

                cv2.rectangle(image_full, (left, bottom - 10), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(image_full, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            
            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image_full, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image_full, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2)
    cv2.imshow("Image", image_full)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()