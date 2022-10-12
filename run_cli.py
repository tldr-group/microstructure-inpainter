import argparse
import os
from pathlib import Path
from src import networks, util
from src.train_rect import RectWorker
from src.train_poly import PolyWorker
from config import Config, ConfigPoly
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
from src.util_cli import *
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def main(mode, tag, coords, path, image_type, shape, wandb):
    """[summary]

    :param mode: [description]
    :type mode: [type]
    :param offline: [description]
    :type offline: [type]
    :param tag: [description]
    :type tag: [type]
    :raises ValueError: [description]
    """
    print("Running in {} mode, tagged {}".format(mode, tag))
    root = Path(__file__).parent
    temp_path = str(root / "data/temp/temp.png")

        
    if shape=='rect':
        # load config and command line arguments
        c = Config(tag, root)
        c.temp_path = temp_path
        c.root = str(root)
        c.data_path = path
        c.mask_coords = tuple(coords)
        c.image_type = image_type
        c.cli = True
        c.wandb = bool(wandb)
        if mode=='train':
            overwrite = util.check_existence(tag, root)
            util.initialise_folders(tag, overwrite, root)
        else:
            overwrite = False
        # Pre-process the data and adjust the nets
        training_imgs, nc = util.preprocess(c.data_path, c.image_type)
        mask, unmasked, img_size, seed, c = util.make_mask(training_imgs, c)
        c.seed_x, c.seed_y = int(seed[0].item()), int(seed[1].item())
        c.lx, c.ly = int(img_size[0].item()), int(img_size[1].item())
        if c.image_type == 'n-phase':
            c.n_phases = nc
        elif c.image_type == 'colour':
            c.n_phases = 3
        else:
            c.n_phases = 1
        
        if mode=='train':
            c.update_params()
            c.save()
        else:
            c.load()

        # Build the nets and initialise worker
        netD, netG = networks.make_nets(c, overwrite)
        worker = RectWorker(c, netG, netD, training_imgs, nc, mask, unmasked)
        worker.verbose = True
        if mode == 'train':
            worker.train(wandb=wandbContainer())
        elif mode == 'generate':
            sp = 'out'
            worker.generate(save_path = sp)
        else:
            raise ValueError("Mode not recognised")

    elif shape == 'poly':
        c = ConfigPoly(tag, root)
        c.data_path = path
        c.root = str(root)
        c.temp_path = temp_path
        c.mask_coords = tuple(coords)
        c.image_type = image_type
        c.cli = True
        c.wandb = bool(wandb)
        x1, x2, y1, y2 = coords
        img = plt.imread(c.data_path)
        if image_type == 'n-phase':
            try:
                h, w = img.shape
            except:
                h, w, _ = img.shape
        else:
            h, w, _ = img.shape
        new_polys = [[(x1,y1), (x1, y2), (x2,y2), (x2, y1)]]
        x, y = np.meshgrid(np.arange(w), np.arange(h)) # make a canvas with coordinates
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x,y)).T
        mask = np.zeros((h,w))
        poly_rects = []
        for poly in new_polys: 
            p = MPath(poly) # make a polygon
            grid = p.contains_points(points)
            mask += grid.reshape(h, w)
            xs, ys = [point[1] for point in poly], [point[0] for point in poly]
            poly_rects.append((np.min(xs), np.min(ys), np.max(xs),np.max(ys)))
        if c.cli:
            # correct offset for rect inpaints
            mask = np.roll(mask,(-1,-1), axis=(0,1))
            mask[y1:y2, x2-1] = 1
        seeds_mask = np.zeros((h,w))
        for x in range(c.l):
            for y in range(c.l):
                seeds_mask += np.roll(np.roll(mask, -x, 0), -y, 1)
        seeds_mask[seeds_mask>1]=1
        real_seeds = np.where(seeds_mask[:-c.l, :-c.l]==0)
        if mode=='train':
            overwrite = util.check_existence(tag, root)
            util.initialise_folders(tag, overwrite, root)
        else:
            overwrite = False
        if c.image_type == 'n-phase':
            c.n_phases = len(np.unique(plt.imread(c.data_path)[...,0]))
        elif c.image_type == 'colour':
            c.n_phases = 3
        else:
            c.n_phases = 1
        c.update_params()
        netD, netG = networks.make_nets(c, overwrite)
        worker = PolyWorker(c, netG, netD, real_seeds, mask, poly_rects, c.frames, overwrite)
        worker.verbose = True
        worker.opt_whilst_train = False
        if mode == 'train':
            worker.train(wandb=wandbContainer())
        elif mode == 'generate':
            worker.generate()
    else:
        raise ValueError("Shape not recognised")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode")
    parser.add_argument("-t", "--tag")
    parser.add_argument('-c', '--coords', nargs='+', type=int, help="Enter coords in order x1 x2 y1 y2", default=[0,64,0,64])
    parser.add_argument('-p', '--path', default='data/nmc.png')
    parser.add_argument('-i', '--image_type', choices=['n-phase', 'colour', 'grayscale'], default='n-phase')
    parser.add_argument('-s', '--shape', choices=['rect', 'poly'], default='rect')
    parser.add_argument('-w', '--wandb', choices=["True", "False"], default="True")

    args = parser.parse_args()
    if args.tag:
        tag = args.tag
    else:
        tag = 'test'
    if args.wandb=='True':
        wandb=True
    elif args.wandb=="False":
        wandb=False

    coords = args.coords
    

    main(args.mode, tag, coords, args.path, args.image_type, args.shape, wandb)
    # main('train', 'case2_test', [220, 380, 220, 380], 'data/nmc-1-cal-greyscale.png', 'grayscale', 'rect')