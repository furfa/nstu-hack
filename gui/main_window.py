
import sys

# import some PyQt5 modules
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QImage, QPixmap 
from PyQt5.QtCore import QTimer

# import Opencv module
import cv2

from ui import *



class MainWindow(QWidget):
    # class constructor

    def __init__(self):
        # call QWidget constructor
        super().__init__()

        self.a = 1


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
        self.log_timer.start(300)


    def logWriter(self):
        self.a += 1
        self.ui.log.setText( str(self.a) )


    # view camera
    def viewCam(self):
        # read image in BGR format
        ret, image = self.cap.read()
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
            self.cap = cv2.VideoCapture(0)
            # start timer
            self.cam_timer.start(50)
            # update control_bt text
            self.ui.control_bt.setText("Stop")
        # if timer is started
        else:
            # stop timer
            self.cam_timer.stop()
            # release video capture
            self.cap.release()
            # update control_bt text
            self.ui.control_bt.setText("Start")


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # create and show mainWindow
    mainWindow = MainWindow()
    mainWindow.show()

    sys.exit(app.exec_())