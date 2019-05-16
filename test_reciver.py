import cv2
from Sockets.SocketReciver import SocketReciver
# cap = cv2.VideoCapture(1)
# ss = SocketSender("localhost")
sr = SocketReciver()

while True:
    cv2.imshow("frem", sr.read())
    print(sr.last_mes)
    cv2.waitKey(1)