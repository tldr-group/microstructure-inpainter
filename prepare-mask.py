import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QFileDialog, QMenuBar, QAction
from PyQt5.QtGui import QIcon, QColor, QBrush, QPainter, QPixmap
from PyQt5.QtCore import QPoint, QRect

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setGeometry(30,30,600,400)
        self.setWindowTitle('Microstructure Inpainter')
        self.painter_widget = PainterWidget(self) 
        self.setCentralWidget(self.painter_widget) 
        self.show()


class PainterWidget(QWidget):
    def __init__(self, parent):
        super(PainterWidget, self).__init__(parent)
        self.image = QPixmap("")
        # self.setGeometry(30,30,600,400)
        self.resize(self.image.width(), self.image.height())
        self.begin = QPoint()
        self.end = QPoint()

        loadAct = QAction('Load', self)
        loadAct.setStatusTip('Load new image from file')
        loadAct.triggered.connect(self.onLoadClick)
        loader = parent.addToolBar('&Load')
        loader.addAction(loadAct)

        trainAct = QAction('Train', self)
        trainAct.setStatusTip('Train network')
        trainAct.triggered.connect(self.onTrainClick)
        trainer = parent.addToolBar('&Train')
        trainer.addAction(trainAct)
        
        self.show()


    def setPixmap(self, fp):
        self.image = QPixmap(fp)
        # self.setGeometry(30,30,600,400)
        # self.resize(self.image.width(), self.image.height())
        self.show()


    def openFileNamesDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self,"QFileDialog.getOpenFileNames()", "","All Files (*);;Python Files (*.py)", options=options)
        if files:
            self.setPixmap(files[0])

    def paintEvent(self, event):
        qp = QPainter(self)
        qp.drawPixmap(self.rect(), self.image)
        br = QBrush(QColor(100, 10, 10, 10))  
        qp.setBrush(br)   
        qp.drawRect(QRect(self.begin, self.end))       

    def mousePressEvent(self, event):
        self.begin = event.pos()
        self.end = event.pos()
        self.update()

    def mouseMoveEvent(self, event):
        self.end = event.pos()
        self.update()

    def onLoadClick(self, event):
        self.openFileNamesDialog()
        self.show()
    
    def onTrainClick(self, event):
        print(self.begin.x(), self.end.x())

def main():

    app = QApplication(sys.argv)
    window = MainWindow()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()