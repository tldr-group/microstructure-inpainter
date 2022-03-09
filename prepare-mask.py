import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QFileDialog, QMenuBar, QAction, QComboBox
from PyQt5.QtGui import QIcon, QColor, QBrush, QPainter, QPixmap, QPolygonF, QPen
from PyQt5.QtCore import QPoint, QRect, QPointF

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
        self.shape = 'rect'
        self.poly = []
        self.old_polys = []
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

        self.selectorBox = QComboBox()
        self.selectorBox.insertItems(1,['rectangle', 'poly'])
        selector = parent.addToolBar("Selector")
        selector.addWidget(self.selectorBox)
        self.selectorBox.activated[str].connect(self.onShapeSelected)
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
        pen = QPen(QColor(0, 0, 0, 255), 3)
        qp.setBrush(br)
        qp.setPen(pen)

        if self.shape=='rect':  
            qp.drawRect(QRect(self.begin, self.end))     
        else:
            for p in self.old_polys:
                qp.drawPolygon(self.createPoly(p))
            qp.drawPolygon(self.createPoly(self.poly))
  
    def createPoly(self, polypoints):
        polygon = QPolygonF()                                                     
        for point in polypoints:
            polygon.append(QPointF(point.x(), point.y()))  
        return polygon

    def mousePressEvent(self, event):
        self.begin = event.pos()
        self.end = event.pos()
        if self.shape == 'poly':
            if len(self.poly)==0:
                self.poly.append(self.begin)
            self.poly.append(self.end)
            self.setMouseTracking(True)
        self.update()

    def mouseDoubleClickEvent(self,event):
        if self.shape == 'poly':
            if len(self.poly)==0:
                self.poly.append(self.begin)
            self.poly.append(self.end)
            self.setMouseTracking(False)
            self.old_polys.append(self.poly)
            self.poly = []
        self.update()
        
    def mouseMoveEvent(self, event):
        self.end = event.pos()
        if self.shape == 'poly':
            try:
                self.poly[-1] = self.end
            except:
                pass
        self.update()

    def onLoadClick(self, event):
        self.openFileNamesDialog()
        self.show()

    def onShapeSelected(self, event):
        self.begin = QPoint()
        self.end = QPoint()
        self.poly = []
        self.old_polys = []
        self.shape = self.selectorBox.currentText()
        print(self.shape)
        self.update()

    def onTrainClick(self, event):
        print(self.begin.x(), self.end.x())

def main():

    app = QApplication(sys.argv)
    window = MainWindow()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()