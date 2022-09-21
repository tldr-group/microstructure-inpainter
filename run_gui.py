import shutil
import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QFileDialog,QHBoxLayout, QToolBar, QAction, QComboBox, QLabel, QDialog, QScrollArea
from PyQt5.QtGui import QFontDatabase, QColor, QBrush, QPainter, QPixmap, QPolygonF, QPen
from PyQt5.QtCore import QDir, QPoint, QRect, QPointF, QThread, QTimeLine, QCoreApplication, QProcess, Qt
import matplotlib.pyplot as plt
from src.train_poly import PolyWorker
from src.train_rect import RectWorker
from config import Config, ConfigPoly
from src.networks import make_nets
import src.util as util
from matplotlib.path import Path
import numpy as np
import os

class MainWindow(QMainWindow):
    def __init__(self, primaryScreen):
        super().__init__()
        self.primaryScreen = primaryScreen
        self.initUI()
        self.initToolbar()
        self.connectToolbar()
        self.resizeWindow()

    def resizeWindow(self):
        padding = 40
        im_w, im_h = self.painter.image.width(), self.painter.image.height()
        sc_w, sc_h = self.primaryScreen.availableGeometry().width(), self.primaryScreen.availableGeometry().height()
        tb_w, tb_h = self.mainToolbar.width(), self.mainToolbar.height()
        tb_h = self.mainToolbar.widgetForAction(self.mainToolbar.stopBtn).height() + padding
        if im_w > sc_w:
            w = sc_w
        elif tb_w>im_w:
            w = tb_w
        else:
            w = im_w
        
        if im_h+tb_h > sc_h:
            h = sc_h
        else:
            h = tb_h+im_h
        self.h, self.w = h, w
        self.contents.setMinimumWidth(im_w+5)
        self.contents.setMinimumHeight(im_h+5)
        self.setGeometry(30,30,w,h)

    def initUI(self):
        
        self.setWindowTitle('Microstructure Inpainter')
        self.painter = PainterWidget(self) 
        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        
        self.contents = QWidget()
        self.scroll.setWidget(self.contents)
        self.layout = QHBoxLayout(self.contents)

        self.layout.addWidget(self.painter)
        
        self.setCentralWidget(self.scroll)
        self.show()
    
    def initToolbar(self):
        self.mainToolbar = QToolBar("Main")
        self.addToolBar(self.mainToolbar)

        self.mainToolbar.loadAct = QAction('Load', self)
        self.mainToolbar.loadAct.setStatusTip('Load new image from file')
        # loader = self.addToolBar('&Load')
        self.mainToolbar.addAction(self.mainToolbar.loadAct)

        self.mainToolbar.saveAct = QAction('Save', self)
        self.mainToolbar.saveAct.setStatusTip('Save inpainted image')
        # save = self.addToolBar('&Save')
        self.mainToolbar.addAction(self.mainToolbar.saveAct)

        self.mainToolbar.restartAct = QAction('Restart', self)
        self.mainToolbar.restartAct.setStatusTip('Restart GUI')
        self.mainToolbar.addAction(self.mainToolbar.restartAct)

        self.mainToolbar.trainBtn = QAction('Train', self)
        self.mainToolbar.trainBtn.setStatusTip('Train network')
        self.mainToolbar.trainBtn.triggered.connect(self.painter.onTrainClick)
        self.mainToolbar.addAction(self.mainToolbar.trainBtn)

        self.mainToolbar.stopBtn = QAction("Stop", self)
        self.mainToolbar.addAction(self.mainToolbar.stopBtn)
        self.mainToolbar.stopBtn.setVisible(False)

        self.mainToolbar.generateBtn = QAction("Generate", self)
        self.mainToolbar.addAction(self.mainToolbar.generateBtn)
        self.mainToolbar.generateBtn.setVisible(False)

        self.mainToolbar.ImageTypeBox = QComboBox()
        self.mainToolbar.ImageTypeBox.insertItems(1,['n-phase', 'colour', 'grayscale'])
        self.mainToolbar.addWidget(self.mainToolbar.ImageTypeBox)

        self.mainToolbar.selectorBox = QComboBox()
        self.mainToolbar.selectorBox.insertItems(1,['rectangle', 'poly'])
        self.mainToolbar.addWidget(self.mainToolbar.selectorBox)

        # self.mainToolbar.addToolBarBreak()

        self.mainToolbar.borderAct = QAction('Border', self)
        self.mainToolbar.borderAct.setStatusTip('Toggle patch border')
        self.mainToolbar.addAction(self.mainToolbar.borderAct)

        self.mainToolbar.addWidget(self.painter.step_label)

        self.resizeWindow()
    
    def connectToolbar(self):
        self.mainToolbar.loadAct.triggered.connect(self.painter.onLoadClick)
        self.mainToolbar.saveAct.triggered.connect(self.painter.onSaveClick)
        self.mainToolbar.restartAct.triggered.connect(self.painter.restart)
        self.mainToolbar.stopBtn.triggered.connect(self.painter.stop_train)
        self.mainToolbar.generateBtn.triggered.connect(self.painter.generateInpaint)
        self.mainToolbar.ImageTypeBox.activated[str].connect(self.painter.onImageTypeSelected)
        self.mainToolbar.selectorBox.activated[str].connect(self.painter.onShapeSelected)
        self.mainToolbar.borderAct.triggered.connect(self.painter.onBorderClick)
        

class PainterWidget(QWidget):
    def __init__(self, parent):
        super(PainterWidget, self).__init__(parent)

        self.parent = parent
        self.image = QPixmap("data/example_inpainting.png")
        self.img_path = "data/example_inpainting.png"
        self.shape = 'rect'
        self.image_type = 'n-phase'
        self.tag = 'test_run'
        self.poly = []
        self.old_polys = []
        self.border = True
        self.frames = 100
        self.begin = QPoint()
        self.end = QPoint()
        self.step_label = QLabel('Iter: 0, Epoch: 0, MSE: 0')
        self.training = False
        self.generate = False


        timeLine = QTimeLine(self.frames * 100, self)
        timeLine.setFrameRange(0, self.frames - 1)
        timeLine.frameChanged[int].connect(self.show_next_img)
        self.timeline = timeLine

        self.show()
    
    def restart(self):
        QCoreApplication.quit()
        status = QProcess.startDetached(sys.executable, sys.argv)
    def show_next_img(self, i):
        self.setPixmap(f"data/temp/temp{i}.png")
        
    def onImageTypeSelected(self, event):
        self.image_type = self.parent.mainToolbar.ImageTypeBox.currentText()

    def setPixmap(self, fp):
        self.image = QPixmap(fp)
        self.parent.resizeWindow()
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
        self.shape = self.parent.selectorBox.currentText()
        self.update()

    def onTrainClick(self, event):
        self.training = True
        self.generate = False

        self.parent.mainToolbar.stopBtn.setVisible(True)
        self.parent.mainToolbar.generateBtn.setVisible(False)
        self.parent.mainToolbar.trainBtn.setVisible(False)
        tag = self.tag
        try:
            if self.shape=='rect':
                # transform relative to image
                x1, x2, y1, y2 = self.begin.x(), self.end.x(), self.begin.y(), self.end.y()
                
                # get relative coordinates
                r = self.image.rect()
                
                w = self.frameGeometry().width()
                h = self.frameGeometry().height()
                x1 = int(x1*r.width()/w)
                x2 = int(x2*r.width()/w)
                y1 = int(y1*r.height()/h)
                y2 = int(y2*r.height()/h)

                c = Config(tag)
                c.wandb = False
                c.cli = False
                c.data_path = self.img_path
                c.mask_coords = (x1,x2,y1,y2)
                # overwrite = util.check_existence(tag)
                overwrite = True
                util.initialise_folders(tag, overwrite)
                training_imgs, nc = util.preprocess(c.data_path, self.image_type)
                mask, unmasked, img_size, seed, c = util.make_mask(training_imgs, c)
                c.seed_x, c.seed_y = int(seed[0].item()), int(seed[1].item())
                c.lx, c.ly = int(img_size[0].item()), int(img_size[1].item())
                # Use dl to update discrimantor network structure
                c.image_type = self.image_type
                if self.image_type == 'n-phase':
                    c.n_phases = nc
                elif self.image_type == 'colour':
                    c.n_phases = 3
                else:
                    c.n_phases = 1
                c.update_params()
                c.save()
                netD, netG = make_nets(c, overwrite)
                self.worker = RectWorker(c, netG, netD, training_imgs, nc, mask, unmasked)
                
            elif self.shape=='poly':
                if len(self.old_polys) ==0:
                    self.training=False
                c = ConfigPoly(tag)
                c.wandb = False
                c.cli = False
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
                netD, netG = make_nets(c, overwrite)
                
                print(f'training with {c.n_phases} channels using image type {self.image_type} and net type conv resize')
                self.worker = PolyWorker(c, netG, netD, real_seeds, mask, poly_rects, self.frames, overwrite)

                
                
            self.thread = QThread()
            # Step 3: Create a worker object
            # Step 4: Move worker to the thread
            self.worker.painter = self
            self.worker.moveToThread(self.thread)
            # Step 5: Connect signals and slots
            self.thread.started.connect(self.worker.train)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.worker.finished.connect(self.stop_train)
            self.thread.finished.connect(self.thread.deleteLater)
            self.worker.progress.connect(self.progress)
            # Step 6: Start the thread
            self.thread.start()
        except Exception as e:
            dialog = QDialog(self)
            d_label = QLabel(f'Training could not be started because: {e}',dialog)
            dialog.setWindowTitle("Training not started")
            dialog.show()
            dialog.adjustSize()
            dialog.exec_()
            self.training = False
            self.parent.mainToolbar.stopBtn.setVisible(False)
    
    def stop_train(self):
        self.training = False
        self.parent.mainToolbar.stopBtn.setVisible(False)
        self.parent.mainToolbar.generateBtn.setVisible(True)


    def progress(self, i, t, mse, wass):
        self.step_label.setText(f'Iter: {i}, Time: {t}, MSE: {mse:.2g}, Wass: {wass:.2g}')
        if self.shape=='poly':
            self.timeline.start()
        else:
            self.image = QPixmap("data/temp/temp.png")

    def generateInpaint(self):
        tag = self.tag
        overwrite = False
        if self.shape == 'rect':
            c = Config(tag)
            c.load()
            original_img, nc = util.preprocess('data/temp/temp.png', self.image_type)
            netD, netG = make_nets(c, overwrite)
            self.worker = RectWorker(c, netG, netD, original_img, nc)
            self.worker.verbose = False
            self.worker.generate()
            self.image = QPixmap("data/temp/temp.png")
            self.update()
        elif self.shape == 'poly':
            if len(self.old_polys) ==0:
                    self.training=False
            c = ConfigPoly(tag)
            c.wandb = False
            c.cli = False
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
            if self.image_type == 'n-phase':
                c.n_phases = len(np.unique(plt.imread(c.data_path)[...,0]))
                c.conv_resize=True
                
            elif self.image_type == 'colour':
                c.n_phases = 3
                c.conv_resize = True
            else:
                c.n_phases = 1
            c.image_type = self.image_type
            netD, netG = make_nets(c, overwrite)
            self.worker = PolyWorker(c, netG, netD, real_seeds, mask, poly_rects, self.frames, overwrite)
            self.worker.verbose = False
            self.worker.generate(opt_iters=1000)
            self.timeline.start()
            self.update()
        
def clear_temp():
    folder = 'data/temp'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def main():

    app = QApplication(sys.argv)
    qss="style.qss"
    with open(qss,"r") as fh:
        app.setStyleSheet(fh.read())
    dir_ = QDir("Roboto")
    _id = QFontDatabase.addApplicationFont("assets/ariblk.ttf")
    clear_temp()
    window = MainWindow(app.primaryScreen())

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()