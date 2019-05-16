import cv2
from Sockets.SocketReciver import SocketReciver
# cap = cv2.VideoCapture(1)
# ss = SocketSender("localhost")
sr = SocketReciver()

while True:
    cv2.imshow("frem", sr.read())
    l_m = sr.read_mes()
    if l_m != None:
        print(l_m[0])
    cv2.waitKey(1)