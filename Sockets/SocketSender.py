import socket
import cv2
import numpy as np 
import time
import json
import struct
import zlib
import pickle
from imagezmq import imagezmq
import zmq
import base64
class SocketSender:
    def __init__(self, addr='tcp://127.0.0.1'):
        # self.sock = socket.socket()
        # self.sock.connect((addr, 9090))
        # self.sock_img = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.sock_img.connect((addr, 9095))
        # context = zmq.Context()
        self.sender = imagezmq.ImageSender('tcp://127.0.0.1:5555')
        self.socket = self.sender.zmq_context.socket(zmq.REQ)
        # self.socket.connect(addr+"%s" % 5588)
        self.socket.connect("tcp://localhost:%s" % 5588)
        # self.image_sender = context.socket(zmq.REQ)
        # self.image_sender.connect('tcp://localhost:5555')
        

    def send_image(self, cam, image):
        image = cv2.resize(image, (320, 240))
        # res, image = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        # data = pickle.dumps(image, 0)
        # size = len(data)
        # print(size)
        # print(struct.calcsize(struct.pack(">L", size)))
        # print("{}: {}".format(img_counter, size))
        # self.sock_img.sendall(struct.pack(">L", size) + data)
        # frame = cv2.resize(image, (640, 480))       # resize the frame
        # self.image_sender.send(base64.b64encode(frame))
        self.sender.send_image(cam, image)  

    def send_action(self, cam, obj):
        # action_timed = {"time":time.asctime(time.localtime()),"cam":str(cam), "action":action}
        # self.data["actions"].append(action_timed)
        # self.data["action_count"] = len(self.data["actions"])
        self.socket.send_json(obj)
        message = self.socket.recv()

    # def close(self):
    #     self.sock.close()
    #     self.sock_img.close()