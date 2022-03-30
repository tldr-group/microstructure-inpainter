import argparse
import os
import tifffile
import torch
from src import networks, util
from src.train_rect import RectWorker
from config import Config
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def main(mode, tag, coords, path, image_type, shape):
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

    # Initialise Config object
    c = Config(tag)
    c.data_path = path
    c.mask_coords = tuple(coords)
    c.image_type = image_type
    c.cli = True

    overwrite = util.check_existence(tag)
    util.initialise_folders(tag, overwrite)
        
    if shape=='rect':
        training_imgs, nc = util.preprocess(c.data_path, c.image_type)
        mask, unmasked, dl, img_size, seed, c = util.make_mask(training_imgs, c)
        c.seed_x, c.seed_y = int(seed[0].item()), int(seed[1].item())
        c.dl, c.lx, c.ly = dl, int(img_size[0].item()), int(img_size[1].item())
        if c.image_type == 'n-phase':
            c.n_phases = nc
        elif c.image_type == 'colour':
            c.n_phases = 3
        else:
            c.n_phases = 1
        c = util.update_discriminator(c)
        c.update_params()
        c.save()
        netD, netG = networks.make_nets_rect(c, overwrite)
        worker = RectWorker(c, netG, netD, training_imgs, nc, mask, unmasked)
        if mode == 'train':
            worker.train()
        elif mode == 'generate':
            worker.generate()



    else:
        raise ValueError("Mode not recognised")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode")
    parser.add_argument("-t", "--tag")
    parser.add_argument('-c', '--coords', nargs='+', type=int, help="Enter coords in order x1 x2 y1 y2")
    parser.add_argument('-p', '--path')
    parser.add_argument('-i', '--image_type', choices=['n-phase', 'colour', 'grayscale'])
    parser.add_argument('-s', '--shape', choices=['rect', 'poly'])

    args = parser.parse_args()
    if args.tag:
        tag = args.tag
    else:
        tag = 'test'

    coords = args.coords
    

    main(args.mode, tag, coords, args.path, args.image_type, args.shape)
    # main('train', True, 'test')