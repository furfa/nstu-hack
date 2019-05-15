# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui_main_window.ui'
#
# Created by: PyQt5 UI code generator 5.12.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1017, 843)
        Form.setStyleSheet("background-color: rgb(0, 0, 0);\n"
"color: rgb(255, 255, 255);")
        self.horizontalLayout = QtWidgets.QHBoxLayout(Form)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.control_bt = QtWidgets.QPushButton(Form)
        self.control_bt.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.control_bt.setStyleSheet("background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 rgba(255, 0, 0, 255), stop:0.166 rgba(255, 255, 0, 255), stop:0.333 rgba(0, 255, 0, 255), stop:0.5 rgba(0, 255, 255, 255), stop:0.666 rgba(0, 0, 255, 255), stop:0.833 rgba(255, 0, 255, 255), stop:1 rgba(255, 0, 0, 255));\n"
"\n"
"color: rgb(206, 92, 0);\n"
"font: italic 18pt \"ori1Uni\";\n"
"\n"
"border: 1px solid red;\n"
"border-radius: 4px;")
        self.control_bt.setObjectName("control_bt")
        self.verticalLayout.addWidget(self.control_bt)
        self.image_label = QtWidgets.QLabel(Form)
        self.image_label.setStyleSheet("border: 1px solid red;")
        self.image_label.setObjectName("image_label")
        self.verticalLayout.addWidget(self.image_label)
        self.log = QtWidgets.QLabel(Form)
        self.log.setEnabled(True)
        self.log.setMaximumSize(QtCore.QSize(1000, 392))
        font = QtGui.QFont()
        font.setFamily("Font Awesome 5 Brands")
        font.setPointSize(29)
        font.setBold(True)
        font.setWeight(75)
        self.log.setFont(font)
        self.log.setContextMenuPolicy(QtCore.Qt.NoContextMenu)
        self.log.setObjectName("log")
        self.verticalLayout.addWidget(self.log)
        self.horizontalLayout.addLayout(self.verticalLayout)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Cam view"))
        self.control_bt.setText(_translate("Form", "Start"))
        self.image_label.setText(_translate("Form", "TextLabel"))
        self.log.setText(_translate("Form", "TextLabel"))


