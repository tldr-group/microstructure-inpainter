import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import torch
from config import Config, ConfigPoly
from src import networks, util
from src.train_poly import PolyWorker
from src.train_rect import RectWorker
from scipy.stats import ttest_ind, ks_2samp
from cycler import cycler

global_colours = ['#253659', '#04BF9D', '#F27457', '#BF665E']
prop_cycler = cycler(color=global_colours)

def main(tag1, tag2, legend=['G optimisation', 'z optimisation'], generate=False, metric_compare=False, mse_plot=False, load=False, wass_plot=False):

    # Plot analytics

    if mse_plot:

        x = 'time'
        if x == 'iters':
            x_lab = "Iterations"
        elif x == 'time':
            x_lab = "Time / s"
        max = 60*60*5
        df1 = pd.read_pickle(f'runs/{tag1}/metrics.pkl')
        df2 = pd.read_pickle(f'runs/{tag2}/metrics.pkl')

        df1 = df1[df1[x]<=max]
        df2 = df2[df2[x]<=max]

        plt.figure()
        plt.rc('axes', prop_cycle=prop_cycler)
        plt.semilogy(df1[x], df1['mse'], label=legend[0])
        plt.semilogy(df2[x], df2['mse'], label=legend[1])
        # plt.ylim(0,0.1)
        plt.xlabel(x_lab)
        plt.ylabel("MSE")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'analysis/mse_{tag1}_{tag2}_{x}.eps', transparent=True)
        plt.close()
    
    if wass_plot:

        x = 'iters'
        if x == 'iters':
            x_lab = "Iterations"
        elif x == 'time':
            x_lab = "Time / s"
        max = 60*60*5
        df1 = pd.read_pickle(f'runs/{tag1}/metrics.pkl')
        df2 = pd.read_pickle(f'runs/{tag2}/metrics.pkl')

        df1 = df1[df1[x]<=max]
        df2 = df2[df2[x]<=max]

        plt.figure()
        plt.rc('axes', prop_cycle=prop_cycler)
        plt.semilogy(df1[x], df1['wass'], label=legend[0])
        plt.semilogy(df2[x], df2['wass'], label=legend[1])
        # plt.ylim(0,0.1)
        plt.xlabel(x_lab)
        plt.ylabel("MSE")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'analysis/wass_{tag1}_{tag2}_{x}.png')
        plt.close()

    c = Config(tag1)
    c.load()

    # Generate examples
    y1, y2, x1, x2 = c.mask_coords
    c.mask_coords = (x1, x2, y1, y2)
    path = c.data_path
    image_type = c.image_type
    # metric comparison params
    samples = 128
    lx, ly = x2-x1, y2-y1
    img, n_phases = util.preprocess(path, image_type)
    img = (img/img.max())
    if generate:
        # save ground truth
        # img = plt.imread(path)
        
        fig, ax = plt.subplots()

        # ax.imshow(np.stack([img for i in range(3)], -1))
        if image_type=='grayscale':
            ax.imshow(util.post_process(img.unsqueeze(0),c)[0].permute(1,2,0), cmap='gray')
            rect_col = '#CC2825'
        else:
            ax.imshow(util.post_process(img.unsqueeze(0),c)[0].permute(1,2,0))
            rect_col='#CC2825'
        rect = Rectangle((x1, y1),x2-x1,y2-y1,linewidth=1,ls='--', edgecolor=rect_col,facecolor='none')
        ax.add_patch(rect)
        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(f'analysis/{tag1}_{tag2}_GT.png', transparent=True)

        
    real_list = []
    if metric_compare and not load:
        for i in range(samples):
            x = np.random.randint(0,img.shape[1]-lx)
            y = np.random.randint(0,img.shape[2]-ly)
            im = img[:, x:x+lx, y:y+ly]
            vfs = ()
            for p in range(n_phases):
                vfs = vfs + (im[p].mean().item(),)
            real_list.append(vfs)
                
            
        

    # RECT

    c = Config(tag1)
    c.load()
    c.mask_coords = (x1, x2, y1, y2)
    overwrite = False
    training_imgs, nc = util.preprocess(c.data_path, c.image_type)
    mask, unmasked, dl, img_size, seed, c = util.make_mask(training_imgs, c)
    netD, netG = networks.make_nets_rect(c, overwrite)
    worker = RectWorker(c, netG, netD, training_imgs, nc, mask, unmasked)
    if generate:
        for i in range(5):
            worker.generate(save_path=f'analysis/{tag1}_{i}', border=True)
    if metric_compare and not load:
        rect_list = []
        for i in range(samples):
            im = worker.generate()[0].cpu()
            im = (torch.argmax(im, dim=0))
            x,y = im.shape
            im = im[(x-lx)//2:(x+lx)//2,(y-ly)//2:(y+ly)//2,]
            ph = np.unique(im)
            vfs = ()
            for p in ph:
                i = im==p
                vfs = vfs + (i.float().mean().item(),)
            rect_list.append(vfs)


    # POLY

    c = ConfigPoly(tag2)
    c.data_path = path
    c.mask_coords = tuple([x1,x2,y1,y2])
    c.image_type = image_type
    c.cli = True
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
    overwrite = False
    if c.image_type == 'n-phase':
        c.n_phases = len(np.unique(plt.imread(c.data_path)[...,0]))
        c.conv_resize=True
        
    elif c.image_type == 'colour':
        c.n_phases = 3
        c.conv_resize = True
    else:
        c.n_phases = 1
    c.image_type = c.image_type
    netD, netG = networks.make_nets_poly(c, overwrite)
    worker = PolyWorker(c, netG, netD, real_seeds, mask, poly_rects, c.frames, overwrite)
    if generate:
        for i in range(5):
            worker.generate(save_path=f'analysis/{tag2}_{i}', border=True)
    if metric_compare:
        if not load:
            poly_list = [] = []
            for i in range(samples):
                im = worker.generate()[0].detach().cpu()
                im = (torch.argmax(im, dim=0))
                x,y = im.shape
                im = im[(x-lx)//2:(x+lx)//2,(y-ly)//2:(y+ly)//2,]
                ph = np.unique(im)
                vfs = ()
                for p in ph:
                    i = im==p
                    vfs = vfs + (i.float().mean().item(),)
                poly_list.append(vfs)


            vfs_dict = {}
            type_list = ['Real', 'G opt', 'z opt']
            labels = []
            data = []
            colours_list = ['red', 'blue', 'green']
            colours = []
            for i, ls in enumerate([real_list, rect_list, poly_list]):
                label = type_list[i]
                vfs_dict[label] = {}
                for j, ph in enumerate(['Phase 1', "Phase 2"]):
                    vfs_dict[label][ph] = [a[j] for a in ls]
                    labels.append(label+' '+ph)
                    data.append([a[j] for a in ls])
                    colours.append(colours_list[i])
            with open(f'analysis/vfs_{tag1}_{tag2}_analysis.json', 'w') as fp:
                        json.dump(vfs_dict, fp)
        if load:
            data = []
            labels = []
            colours = []
        plot_vfs(tag1, tag2, data=data, labels=labels, colours=colours, load=load)
        
        
def plot_vfs(tag1, tag2, data=None, labels=None, colours=None, load=True):
    if load:
        with open(f'analysis/vfs_{tag1}_{tag2}_analysis.json', 'r') as fp:
            loaded = json.load(fp)
        colours = []
        labels = []
        colours_list = global_colours
        data = []
        phases = []
        key_list = loaded[list(loaded.keys())[0]].keys()
        for j, k2 in enumerate(key_list):
            for i, k1 in enumerate(loaded.keys()):
                labels.append(k1)
                phases.append(k2)
                colours.append(colours_list[j])
                data.append(loaded[k1][k2])
    fig, ax = plt.subplots()
    parts = ax.violinplot(data, showextrema=False, showmeans=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colours[i])
        pc.set_alpha(1)
    
    patches = []
    phs = []
    for p, c in zip(pd.unique(phases),pd.unique(colours)):
        patches.append(mpatches.Patch(color=c, alpha=1))
        phs.append(p)
    ax.legend(patches, phs)
    ax.set_xticks(np.arange(1,len(labels)+1))
    ax.set_xticklabels(labels)
    ax.set_ylabel('Volume fraction')
    plt.savefig(f'analysis/{tag1}_{tag2}_vf_violinplot.eps', transparent=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--generate")
    parser.add_argument("-mse")
    parser.add_argument("-wass")
    parser.add_argument("-metric")
    parser.add_argument("-t1", "--tag1")
    parser.add_argument("-t2", "--tag2")
    


    args = parser.parse_args()

    if args.generate=='t':
        generate=True
    else:
        generate=False
    if args.mse=='t':
        mse=True
    else:
        mse=False
    if args.wass=='t':
        wass=True
    else:
        wass=False
    if args.metric=='load':
        metric=True
        load=True
    elif args.metric=='gen':
        metric=True
        load=False
    else:
        metric=False
        load=False

    main(args.tag1, args.tag2, generate=generate, mse_plot=mse, wass_plot=wass, metric_compare=metric, load=load)
    # plot_vfs(tag='case1_rect', load=True)
    # main('train', True, 'test')