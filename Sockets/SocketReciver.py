import socket
import cv2
import numpy as np 
import time
import json
import struct
import zlib
from threading import Thread
import zmq
import base64
from imagezmq import imagezmq


class SocketReciver:
    def __init__(self):
        # self.sock = socket.socket()
        # # self.sock.connect((addr, 9090)) 
        # self.sock_img = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        # self.sock_img.bind(("",9095))
        # self.sock_img.listen(10)
        # self.conn,addr=self.sock_img.accept()
        self.frame = np.zeros((240,320,3), np.uint8)
        # context = zmq.Context()
        self.image_hub = imagezmq.ImageHub()
        self.socket = self.image_hub.zmq_context.socket(zmq.REP)
        self.socket.bind("tcp://*:%s" % 5588)


        # self.payload_size = struct.calcsize(">L")
        
        
        # self.image_recv = context.socket(zmq.REP)
        # self.image_recv.bind('tcp://*:5555')
        # self.image_recv.setsockopt_string(zmq.SUBSCRIBE, '')
        self.runing = True
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        t2 = Thread(target=self.update2, args=())
        t2.daemon = True
        t2.start()
        self.last_action = None
        data = b""

    def update2(self):
        while self.runing:
            self.last_action  = self.socket.recv_json()
            self.socket.send_string("reply")
        
    def update(self):
        while self.runing:
            # data = b""
            # frame = self.image_recv.recv_string()
            # frame = np.fromstring(base64.b64decode(frame), dtype=np.uint8)
            image_name, frame = self.image_hub.recv_image()
            self.frame = frame
            self.image_hub.send_reply(b'OK')
            # try:
            #     print(self.payload_size)
            #     while len(data) <=  self.payload_size:
            #         print("Recv: {}".format(len(data)))
            #         data += self.conn.recv(1)
            #         print(data)

            #     packed_msg_size = data[:self.payload_size]
            #     msg_size = struct.unpack(">L", packed_msg_size)[0]
            #     print(msg_size)
            #     while len(data) <= msg_size+self.payload_size:
            #         data += conn.recv(2)
            #     frame_data = data[:msg_size]
            #     data = data[msg_size:]
            #     frame=pickle.loads(frame_data, fix_imports=True, encoding="bytes")
            #     frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
            #     frame = cv2.resize(frame, (img_size[1], img_size[0]))
            #     self.est = True
                
            #     self.frame = frame

            # except:
            #     # cv2.destroyAllWindows()
            #     self.release()
            #     break
    def read(self):
        # return the frame most recently read
        return self.frame

    def release(self):
        # indicate that the thread should be stopped
        # self.sock.close()
        self.runing = False