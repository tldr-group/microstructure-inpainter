# import tifffile
# import config
# import matplotlib.pyplot as plt

# c = config.Config('test')
# print(c.l)
# data = tifffile.imread(c.data_path)
# print(data.shape)
# mask_factor = 4

# mask = data[data.shape[0]//2-c.l//2:data.shape[0]//2+c.l//2, data.shape[0]//2-c.l//2:data.shape[0]//2+c.l//2, data.shape[0]//2-c.l//2:data.shape[0]//2+c.l//2].copy()
# unmasked = mask.copy()
# mask[c.l//2-c.l//mask_factor:c.l//2+c.l//mask_factor,c.l//2-c.l//mask_factor:c.l//2+c.l//mask_factor,c.l//2-c.l//mask_factor:c.l//2+c.l//mask_factor]=4

# print(mask.shape)

# fig, axs = plt.subplots(nrows=1, ncols=2)
# axs[0].imshow(data[data.shape[0]//2-c.l//2:data.shape[0]//2+c.l//2, data.shape[0]//2-c.l//2:data.shape[0]//2+c.l//2,data.shape[0]//2])
# axs[1].imshow(mask[...,c.l//2])
# plt.savefig('test.png')


# tifffile.imwrite('data/mask.tif', mask)
# tifffile.imwrite('data/unmasked.tif', unmasked)

import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QFileDialog, QMenuBar, QAction
from PyQt5.QtGui import QIcon, QColor, QBrush, QPainter, QPixmap
from PyQt5.QtCore import QPoint, QRect

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.menubar = self.menuBar()
        self.menubar.setNativeMenuBar(False)
        self.setGeometry(30,30,600,400)
        self.setWindowTitle('Microstructure Inpainter')
        self.painter_widget = PainterWidget(self) 
        self.setCentralWidget(self.painter_widget) 
        self.show()


class PainterWidget(QWidget):
    def __init__(self, parent):
        super(PainterWidget, self).__init__(parent)
        self.image = QPixmap("data/underground3.png")
        # self.setGeometry(30,30,600,400)
        self.resize(self.image.width(), self.image.height())
        self.begin = QPoint()
        self.end = QPoint()

        loadAct = QAction('Load', self)
        loadAct.setStatusTip('Load new image from file')
        loadAct.triggered.connect(self.onLoadClick)
        loader = parent.menubar.addMenu('&Load')
        loader.addAction(loadAct)

        trainAct = QAction('Train', self)
        trainAct.setStatusTip('Train network')
        trainAct.triggered.connect(self.onTrainClick)
        trainer = parent.menubar.addMenu('&Train')
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