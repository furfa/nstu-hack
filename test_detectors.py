import cv2
import numpy as np
import face_recognition

from PersonDetectors.YoloPersonDetector import YoloPersonDetetor
from PersonDetectors.SSDPersonDetector import SSDPersonDetetor

# p_detector = YoloPersonDetetor(
#     "./PersonDetectors/yolov3-tiny.cfg", "./PersonDetectors/yolov3-tiny.weights"
# )

# p_detector = YoloPersonDetetor(
#     "./PersonDetectors/yolov3.cfg", "./PersonDetectors/yolov3.weights"
# )

p_detector = SSDPersonDetetor(
    "./PersonDetectors/MobileNetSSD_deploy.prototxt.txt",
    "./PersonDetectors/MobileNetSSD_deploy.caffemodel",
)

# video_capture = cv2.VideoCapture("./../test_out_04.avi")
video_capture = cv2.VideoCapture('./../test_out_03.avi')
# video_capture = cv2.VideoCapture('./../test_out_02.avi')
# video_capture = cv2.VideoCapture(0)
while True:
    ret, frame = video_capture.read()
    image_full = cv2.resize(frame, (0, 0), fx=0.7, fy=0.7)
    image_yolo = cv2.resize(image_full, (0, 0), fx=1, fy=1)
    persons = p_detector.person_locations(image_yolo)
    for i in range(len(persons)):
        # extract the bounding box coordinates
        (x, y) = (persons[i][0], persons[i][1])
        (w, h) = (persons[i][2], persons[i][3])
        # x /= 0.35
        # y /= 0.35
        # w /= 0.35
        # h /= 0.35
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        if w > 50 and h > 50 and y > 0 and x > 0:
            crop = image_full[y : y + h // 3, x : x + w]
            crop = crop[:, :, ::-1]
            face_locations = face_recognition.face_locations(crop)
            # face_locations = fsd.face_locations(crop)
            # face_encodings = face_recognition.face_encodings(crop, face_locations)

            # face_names = []
            # for face_encoding in face_encodings:

            #     matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            #     name = "Unknown"

            #     if True in matches:
            #         first_match_index = matches.index(True)
            #         name = known_face_names[first_match_index]

            #     face_names.append(name)
            # for (top, right, bottom, left), name in zip(face_locations, face_names):
            for (top, right, bottom, left) in face_locations:
                left += x
                top += y
                right += x
                bottom += y

                # cv2.imshow("face", image_full[top: bottom, left:right])
                cv2.rectangle(image_full, (left, top), (right, bottom), (0, 0, 255), 2)
                # font = cv2.FONT_HERSHEY_DUPLEX
                # cv2.putText(image_full, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            # draw a bounding box rectangle and label on the image
            cv2.rectangle(image_full, (x, y), (x + w, y + h), (0, 255, 150), 2)
    cv2.imshow("Image", image_full)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
