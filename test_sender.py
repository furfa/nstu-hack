import cv2
from Sockets.SocketSender import SocketSender
cap = cv2.VideoCapture(1)
ss = SocketSender("localhost")

while True:
    _, img = cap.read()
    ss.send_image(0, img)
    # cv2.waitKey(1)
    ss.send_action(0, ["qwe", "wer"])