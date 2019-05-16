
import sys
sys.path.append("..")

# import some PyQt5 modules
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QImage, QPixmap 
from PyQt5.QtCore import QTimer

from Sockets.SocketReciver import SocketReciver

# import Opencv module
import cv2

from ui import *

from time import sleep


class MainWindow(QWidget):
    # class constructor

    def __init__(self):
        # call QWidget constructor
        super().__init__()

        self.photos_layout = list()

        self.sr = SocketReciver()


        self.ui = Ui_Form()
        self.ui.setupUi(self)

        # create a timer
        self.cam_timer = QTimer()
        self.log_timer = QTimer()

        # set timer timeout callback function
        self.cam_timer.timeout.connect(self.viewCam)
        # set control_bt callback clicked  function
        self.ui.control_bt.clicked.connect(self.controlTimer)

        self.log_timer.timeout.connect(self.logWriter)
        self.log_timer.start(3000)


    def logWriter(self):

        data = self.sr.read_mes()

        self.ui.log.setText( str(data) )

        self.update_recognized_faces(data)

    def update_recognized_faces(self, data):

        if (data == None):
            return

        for i in range( len(data) - len(self.photos_layout) ):
            self.add_photo()


        for label in self.photos_layout:
            
            self.update_photo(label, cv2.imread("2.jpg") )


    def add_photo(self, photo = cv2.imread("default.jpg") ):

        photo_label = QtWidgets.QLabel()
        self.ui.grid.addWidget( photo_label )

        self.update_photo(photo_label, photo)

        self.photos_layout.append(photo_label)

    def update_photo(self, label, image):

        image = cv2.resize(image, (100, 100), )   

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # get image infos
        height, width, channel = image.shape

        step = channel * width
        # create QImage from image
        qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)

        # засовываем пикчу в виджет
        label.setPixmap(QPixmap.fromImage(qImg))


    # view camera
    def viewCam(self):
        # read image in BGR format
        
        #ret, image = self.cap.read()

        image = self.sr.read()


        # convert image to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # get image infos
        height, width, channel = image.shape

        step = channel * width
        # create QImage from image
        qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
        # засовываем пикчу в виджет
        self.ui.image_label.setPixmap(QPixmap.fromImage(qImg))

    # start/stop timer
    def controlTimer(self):
        # if timer is stopped
        if not self.cam_timer.isActive():
            # create video capture
            
            
            #self.cap = cv2.VideoCapture(0)


            # start timer
            self.cam_timer.start(50)
            # update control_bt text
            self.ui.control_bt.setText("Stop")
        # if timer is started
        else:
            # stop timer
            self.cam_timer.stop()
            # release video capture


            #self.cap.release()


            # update control_bt text
            self.ui.control_bt.setText("Start")


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # create and show mainWindow
    mainWindow = MainWindow()
    mainWindow.show()

    sys.exit(app.exec_())