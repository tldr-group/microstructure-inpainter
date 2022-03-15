import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QFileDialog, QAction, QComboBox, QLabel
from PyQt5.QtGui import QColor, QBrush, QPainter, QPixmap, QPolygonF, QPen
from PyQt5.QtCore import QPoint, QRect, QPointF, QThread, QTimeLine
import matplotlib.pyplot as plt
from sympy import re
from src.train_poly import PolyWorker
from src.train_rect import RectWorker
from config import Config, ConfigPoly
from src.networks import make_nets_rect, make_nets_poly
import src.util as util
from matplotlib.path import Path
import numpy as np
import os
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        
        # self.setGeometry(30,30,600,400)
        self.setWindowTitle('Microstructure Inpainter')
        self.painter_widget = PainterWidget(self) 
        self.setCentralWidget(self.painter_widget)
        self.extra_padding = 100
        self.setGeometry(30, 30, self.painter_widget.image.width(), self.painter_widget.image.height()+self.extra_padding)
        self.show()

class PainterWidget(QWidget):
    def __init__(self, parent):
        super(PainterWidget, self).__init__(parent)
        self.parent = parent
        self.image = QPixmap("data/nmc.png")
        self.img_path = "data/nmc.png"
        self.shape = 'rect'
        self.image_type = 'n-phase'
        self.poly = []
        self.old_polys = []
        self.border = True
        self.frames = 100
        self.begin = QPoint()
        self.end = QPoint()
        self.step_label = QLabel('Iter: 0, Epoch: 0, MSE: 0')
        self.training = False
        self.generate = False

        self.stopTrain = QPushButton('Stop', self)
        self.stopTrain.setText("Stop")
        self.stopTrain.move(10,10)
        self.stopTrain.hide()

        self.generateBtn = QPushButton('Generate', self)
        self.generateBtn.setText("Generate")
        self.generateBtn.move(10,10)
        self.generateBtn.hide()
        self.generateBtn.clicked.connect(self.generateInpaint)

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

        borderAct = QAction('Border', self)
        borderAct.setStatusTip('Toggle patch border')
        borderAct.triggered.connect(self.onBorderClick)
        border = parent.addToolBar('&Border')
        border.addAction(borderAct)

        self.ImageTypeBox = QComboBox()
        self.ImageTypeBox.insertItems(1,['colour', 'n-phase', 'grayscale'])
        selector = parent.addToolBar("Image Type")
        selector.addWidget(self.ImageTypeBox)
        self.ImageTypeBox.activated[str].connect(self.onImageTypeSelected)

        self.selectorBox = QComboBox()
        self.selectorBox.insertItems(1,['rectangle', 'poly'])
        selector = parent.addToolBar("Selector")
        selector.addWidget(self.selectorBox)
        self.selectorBox.activated[str].connect(self.onShapeSelected)

        saveAct = QAction('Save', self)
        saveAct.setStatusTip('Save inpainted image')
        saveAct.triggered.connect(self.onSaveClick)
        save = parent.addToolBar('&Save')
        save.addAction(saveAct)

        parent.addToolBarBreak()

        label = parent.addToolBar('Step Label')
        label.addWidget(self.step_label)

        timeLine = QTimeLine(self.frames * 100, self)
        timeLine.setFrameRange(0, self.frames - 1)
        timeLine.frameChanged[int].connect(self.show_next_img)
        self.timeline = timeLine

        self.show()
     
    def show_next_img(self, i):
        self.setPixmap(f"data/temp/temp{i}.png")
        
    def onImageTypeSelected(self, event):
        self.image_type = self.ImageTypeBox.currentText()

    def setPixmap(self, fp):
        self.image = QPixmap(fp)
        self.parent.setGeometry(30, 30, self.image.width(), self.image.height()+self.parent.extra_padding)
        self.update()


    def openFileNamesDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self,"QFileDialog.getOpenFileNames()", "","All Files (*);;Python Files (*.py)", options=options)
        if files:
            self.setPixmap(files[0])
            self.img_path = files[0]

    def onSaveClick(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self, "Save Image", os.getcwd(), "All Files (*)", options=options)
        if fileName:
            img = plt.imread('data/temp/temp.png')
            plt.imsave(fileName, img)
            print(f"Image saved as: {fileName}")

    def paintEvent(self, event):
        qp = QPainter(self)
        self.resize(self.image.width(), self.image.height())
        qp.drawPixmap(self.rect(), self.image)
        br = QBrush(QColor(100, 10, 10, 10))  
        pen = QPen(QColor(0, 0, 0, 255), 1.5)
        qp.setBrush(br)
        qp.setPen(pen)
        if self.border:
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
        if not self.training:
            self.begin = event.pos()
            self.end = event.pos()
            if self.shape == 'poly':
                if len(self.poly)==0:
                    self.poly.append(self.begin)
                self.poly.append(self.end)
                self.setMouseTracking(True)
            self.update()

    def mouseDoubleClickEvent(self,event):
        if not self.training:
            if self.shape == 'poly':
                if len(self.poly)==0:
                    self.poly.append(self.begin)
                self.poly.append(self.end)
                self.setMouseTracking(False)
                self.old_polys.append(self.poly)
                self.poly = []
            self.update()
        
    def mouseMoveEvent(self, event):
        if not self.training:
            self.end = event.pos()
            if self.shape == 'poly':
                try:
                    self.poly[-1] = self.end
                except:
                    pass
            if self.shape == 'rect':
                self.end = ((self.end-self.begin)/16) *16+self.begin
            self.update()

    def onLoadClick(self, event):
        self.openFileNamesDialog()
        self.show()

    def onBorderClick(self, event):
        self.border = 0 if self.border else 1
        self.update()

    def onShapeSelected(self, event):
        self.begin = QPoint()
        self.end = QPoint()
        self.poly = []
        self.old_polys = []
        self.shape = self.selectorBox.currentText()
        self.update()

    def onTrainClick(self, event):
        self.training = True
        self.generate = False

        self.stopTrain.show()
        self.generateBtn.hide()
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

            tag = 'rect'
            c = Config(tag)
            c.data_path = self.img_path
            c.mask_coords = (x1,x2,y1,y2)
            # overwrite = util.check_existence(tag)
            overwrite = True
            util.initialise_folders(tag, overwrite)
            training_imgs, nc = util.preprocess(c.data_path, self.image_type)
            mask, unmasked, dl, img_size, seed, c = util.make_mask(training_imgs, c)
            c.seed_x, c.seed_y = int(seed[0].item()), int(seed[1].item())
            c.dl, c.lx, c.ly = dl, int(img_size[0].item()), int(img_size[1].item())
            # Use dl to update discrimantor network structure
            c.image_type = self.image_type
            if self.image_type == 'n-phase':
                c.n_phases = nc
            elif self.image_type == 'colour':
                c.n_phases = 3
            else:
                c.n_phases = 1
            c = util.update_discriminator(c)
            c.update_params()
            c.save()
            netD, netG = make_nets_rect(c, overwrite)
            self.worker = RectWorker(c, netG, netD, training_imgs, nc, mask, unmasked)
            
        elif self.shape=='poly':
            if len(self.old_polys) ==0:
                self.training=False
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
                xs, ys = [point[1] for point in poly], [point[0] for point in poly]
                poly_rects.append((np.min(xs), np.min(ys), np.max(xs),np.max(ys)))
            seeds_mask = np.zeros((h,w))
            for x in range(c.l):
                for y in range(c.l):
                    seeds_mask += np.roll(np.roll(mask, -x, 0), -y, 1)
            seeds_mask[seeds_mask>1]=1
            real_seeds = np.where(seeds_mask[:-c.l, :-c.l]==0)
            overwrite = True
            util.initialise_folders(tag, overwrite)
            if self.image_type == 'n-phase':
                c.n_phases = len(np.unique(plt.imread(c.data_path)[...,0]))
                c.conv_resize=True
                
            elif self.image_type == 'colour':
                c.n_phases = 3
                c.conv_resize = True
            else:
                c.n_phases = 1
            c.image_type = self.image_type
            netD, netG = make_nets_poly(c, overwrite)
            
            print(f'training with {c.n_phases} channels using image type {self.image_type} and net type conv resize')
            self.worker = PolyWorker(c, netG, netD, real_seeds, mask, poly_rects, self.frames, overwrite)

            # plt.imsave('mask.png', mask)
            # plt.imsave('real_data_seeds.png', rect_mask)

            
            
        self.thread = QThread()
        # Step 3: Create a worker object
        # Step 4: Move worker to the thread
        self.worker.painter = self
        self.worker.moveToThread(self.thread)
        # Step 5: Connect signals and slots
        self.thread.started.connect(self.worker.train)
        self.stopTrain.clicked.connect(lambda: self.worker.stop())
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.finished.connect(self.stop_train)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.progress.connect(self.progress)
        # Step 6: Start the thread
        self.thread.start()
    
    def stop_train(self):
        self.training = False
        self.stopTrain.hide()
        self.generateBtn.show()


    def progress(self, l, e, mse):
        self.step_label.setText(f'Iter: {l}, Epoch: {e}, MSE: {mse:.2g}')
        if self.shape=='poly':
            self.timeline.start()
        else:
            self.image = QPixmap("data/temp/temp.png")

    def generateInpaint(self):
        
        if self.shape == 'rect':
            tag = 'rect'
            c = Config(tag)
            c.load()
            overwrite = False
            original_img, nc = util.preprocess('data/temp/temp.png', self.image_type)
            netD, netG = make_nets_rect(c, overwrite)
            self.worker = RectWorker(c, netG, netD, original_img, nc)
            self.worker.generate()
            self.image = QPixmap("data/temp/temp.png")
            self.update()
        
    

def main():

    app = QApplication(sys.argv)
    qss="style.qss"
    with open(qss,"r") as fh:
        app.setStyleSheet(fh.read())
    window = MainWindow()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()