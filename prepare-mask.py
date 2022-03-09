import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QFileDialog, QMenuBar, QAction
from PyQt5.QtGui import QIcon, QColor, QBrush, QPainter, QPixmap
from PyQt5.QtCore import QPoint, QRect
import matplotlib.pyplot as plt
from src.train import train
from config import Config
from src.networks import make_nets
import src.util as util
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setGeometry(30,30,600,400)
        self.setWindowTitle('Microstructure Inpainter')
        self.painter_widget = PainterWidget(self) 
        self.setCentralWidget(self.painter_widget)
        self.setGeometry(30,30,self.painter_widget.image.width(),self.painter_widget.image.height())
        self.show()


class PainterWidget(QWidget):
    def __init__(self, parent):
        super(PainterWidget, self).__init__(parent)
        self.parent = parent
        self.image = QPixmap("data/Example_ppp.png")
        self.img_path = "data/Example_ppp.png"
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
        self.resize(self.image.width(), self.image.height())
        self.show()


    def openFileNamesDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self,"QFileDialog.getOpenFileNames()", "","All Files (*);;Python Files (*.py)", options=options)
        if files:
            self.setPixmap(files[0])
            self.img_path = files[0]

    def paintEvent(self, event):
        qp = QPainter(self)
        self.resize(self.image.width(), self.image.height())
        qp.drawPixmap(self.rect(), self.image)
        br = QBrush(QColor(100, 10, 10, 10))  
        qp.setBrush(br)   
        qp.drawRect(QRect(self.begin, self.end))
        # self.setGeometry(30,30, self.image.width(), self.image.height())
        

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
        x1, x2, y1, y2 = self.begin.x(), self.end.x(), self.begin.y(), self.end.y()
        # get relative coordinates
        r = self.image.rect()
        w = self.frameGeometry().width()
        h = self.frameGeometry().height()

        x1 = int(x1*r.width()/w)
        x2 = int(x2*r.width()/w)
        y1 = int(y1*r.height()/h)
        y2 = int(y2*r.height()/h)

        tag = 'test'
        c = Config(tag)
        c.data_path = self.img_path
        c.mask_coords = (x1,x2,y1,y2)
        overwrite = util.check_existence(tag)
        util.initialise_folders(tag, overwrite)
        training_imgs, nc = util.preprocess(c.data_path)
        mask, unmasked = util.make_mask(training_imgs, c.mask_coords)
        netD, netG = make_nets(c, overwrite)
        train(c, netG, netD, training_imgs, nc, mask, unmasked, offline=True, overwrite=True)


def main():

    app = QApplication(sys.argv)
    window = MainWindow()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()