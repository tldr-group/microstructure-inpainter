import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QFileDialog, QMenuBar, QAction, QComboBox
from PyQt5.QtGui import QIcon, QColor, QBrush, QPainter, QPixmap, QPolygonF, QPen
from PyQt5.QtCore import QPoint, QRect, QPointF
import matplotlib.pyplot as plt
from sympy import re
from src.train import train_rect, train_poly
from config import Config, ConfigPoly
from src.networks import make_nets_rect
import src.util as util
from matplotlib.path import Path
import numpy as np

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
        self.update()

    def onTrainClick(self, event):
        if self.shape=='rect':
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
            netD, netG = make_nets_rect(c, overwrite)
            train_rect(c, netG, netD, training_imgs, nc, mask, unmasked, offline=True, overwrite=True)
        elif self.shape=='poly':
            tag = 'test'
            c = ConfigPoly(tag)
            c.data_path = self.img_path
            r = self.image.rect()
            w = self.frameGeometry().width()
            h = self.frameGeometry().height()
            rh, rw = r.height(), r.width()
            new_polys = [[(point.x()*rw/w, point.y()*rh/h) for point in poly] for poly in self.old_polys]
            x, y = np.meshgrid(np.arange(w), np.arange(h)) # make a canvas with coordinates
            x, y = x.flatten(), y.flatten()
            points = np.vstack((x,y)).T
            mask = np.zeros((h,w))
            poly_rects = []
            for poly in new_polys: 
                p = Path(poly) # make a polygon
                grid = p.contains_points(points)
                mask += grid.reshape(h, w)
                xs, ys = [point[0] for point in poly], [point[0] for point in poly]
                poly_rects.append((np.min(xs), np.min(ys), np.max(xs),np.max(ys)))
            seeds_mask = np.zeros((h,w))
            for x in range(c.l):
                for y in range(c.l):
                    seeds_mask += np.roll(np.roll(mask, -x, 0), -y, 1)
            seeds_mask[seeds_mask>1]=1
            real_seeds = np.where(seeds_mask[:-c.l, :-c.l]==0)
            overwrite = util.check_existence(tag)
            util.initialise_folders(tag, overwrite)
            netD, netG = make_nets_rect(c, overwrite)
            print('training')
            train_poly(c, netG, netD, real_seeds, mask, poly_rects, offline=True, overwrite=overwrite)
            # plt.imsave('mask.png', mask)
            # plt.imsave('real_data_seeds.png', rect_mask)

def main():

    app = QApplication(sys.argv)
    window = MainWindow()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()