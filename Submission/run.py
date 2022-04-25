from re import A
from PyQt6.QtWidgets import *
from PyQt5.QtGui import * 
from PyQt5.QtWidgets import * 
import sys
from model import VectorSpaceModel
model = VectorSpaceModel()
model.loadORcreateINDEX()
class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Vector Space Model')
        self.resize(500, 350)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.inputField = QLineEdit()
        button = QPushButton('Submit', clicked=self.driverCode)
        self.output = QTextEdit()

        layout.addWidget(self.inputField)
        layout.addWidget(button)
        layout.addWidget(self.output)

    def driverCode(self):
        inputText = self.inputField.text()
        model.getQuery(inputText)
        self.output.setText(str(model.showResult()))

# app = QApplication([])
app = QApplication(sys.argv)
app.setStyleSheet('''
    QWidget {
        font-size: 15px
    }
    QPushButton {
        font-size: 15px
    }
'''
)
window = MyApp()
window.show()
app.exec()