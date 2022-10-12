import torch
from config import ConfigPoly
from src import networks, util
from torch import nn
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from matplotlib.path import Path
from src.train_poly import PolyWorker
import os
import pandas as pd
from scipy.stats import ks_2samp

def border_analysis(img, c, plot=False):
    size = 16
    l = 112
    x1=c.mask_coords[2]-size
    y1=c.mask_coords[0]-size
    gt, _ = util.preprocess(c.data_path, c.image_type)
    gt = gt.permute(1,2,0)[x1:x1+l,y1:y1+l].numpy()

    diffs = []
    data_list = [gt, img]
    name_list = ['Ground truth', 'opt']
    axes_list = [[0],[1]]

    maskl = np.zeros_like(data_list[0])
    maskr = np.zeros_like(data_list[0])
    masku = np.zeros_like(data_list[0])
    maskd = np.zeros_like(data_list[0])
    maskl[size:-size,size:size+1] = 1
    maskd[-size-1:-size, size:-size] = 1
    masku[size:size+1, size:-size] = 1
    maskr[size:-size, -size-1:-size] = 1
    mask = maskl + maskr + masku + maskd
    mask = mask>0
    if plot:
        fig = plt.figure()
        gs = GridSpec(1,3)
        for d, a, n in zip(data_list, axes_list, name_list):
            if len(a)>1:
                x,y = a
                ax = fig.add_subplot(gs[x,y])
            else:
                ax = fig.add_subplot(gs[:,a[0]])
            ax.imshow(d, cmap='gray')
            ax.set_title(n)
            ax.set_axis_off()
            fig.savefig(f'analysis/border_analysis.png')

    
    for j, (d, a, n) in enumerate(zip(data_list, axes_list, name_list)):
        dl = np.roll(d, 1, axis=1)
        dd = np.roll(d, -1, axis=0)
        dr = np.roll(d, -1, axis=1)
        du = np.roll(d, 1, axis=0)
        diffl = ((d-dl)**2)[maskl==1]
        diffd = ((d-dd)**2)[maskd==1]
        diffr = ((d-dr)**2)[maskr==1]
        diffu = ((d-du)**2)[masku==1]
        diff = np.concatenate([diffl, diffd, diffr, diffu])
        diffs.append(diff)

    gt_diff = ((np.roll(data_list[0],1, axis=1)[1:-1,1:-1]-data_list[0][1:-1,1:-1])**2).flatten()

    # MSE hist
    # plt.figure()
    # for x, n in zip(diffs, name_list):
    #     plt.hist(x, label=n, alpha=0.5)
    # plt.legend()
    # plt.xlabel('MSE')
    # plt.ylabel('Freq')
    # plt.tight_layout()
    # plt.savefig('analysis/mse_hist.png')


    df = pd.DataFrame({'Name': name_list,
                        'Mean MSE': [f'{np.mean(t):.2g} ± {np.std(t):.2g}' for t in diffs],
                        'KS value': [f"{ks_2samp(t, gt_diff)[0]:.2g}" for t in diffs],
                        'p': [f"{ks_2samp(t, gt_diff)[-1]:.2g}" for t in diffs]})

    df.index = df['Name']
    df.drop("Name", axis=1, inplace=True)
    print(df.head())
    return df

load = True
if load:
    fls = os.listdir('analysis')
    fls = [f for f in fls if f[0].isdigit()]
    w = 4
    h = len(fls)//w
    print(len(fls))
    
    fig = plt.figure(figsize=(24,24))
    gs = GridSpec(w,h,hspace=0,wspace=0)
    count = 0
    for i in range(w):
        for j in range(h):
            ax = fig.add_subplot(gs[i,j])
            d = plt.imread(f'analysis/{fls[count]}')[10:-10,50:-50]
            ax.imshow(d)
            plt.axis('off')
            count += 1
    plt.tight_layout()
    plt.savefig('analysis/hyperparams_opt_poly.png')
else:
    c = ConfigPoly('case_2_poly')
    c.load()
    # c.data_path = path
    # c.mask_coords = tuple([x1,x2,y1,y2])
    # c.image_type = image_type
    # c.cli = True
    x1,x2,y1,y2 = c.mask_coords
    image_type = c.image_type
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

    saves_list = [0,9,99,999,9999]
    device = torch.device(worker.c.device_name if(
        torch.cuda.is_available() and worker.c.ngpu > 0) else "cpu")
    netG = worker.netG().to(device)
    netD = worker.netD().to(device)
    if ('cuda' in str(device)) and (worker.c.ngpu > 1):
        netD = (nn.DataParallel(netD, list(range(worker.c.ngpu)))).to(device)
        netG = nn.DataParallel(netG, list(range(worker.c.ngpu))).to(device)
    netG.load_state_dict(torch.load(f"{worker.c.path}/Gen.pt"))
    netD.load_state_dict(torch.load(f"{worker.c.path}/Disc.pt"))

    img = util.preprocess(worker.c.data_path, worker.c.image_type)[0]

    for rect in worker.poly_rects:
        x0, y0, x1, y1 = (int(i) for i in rect)
        w, h = x1-x0, y1-y0
        w_init, h_init = w,h
        # x1 += 32 - w%32
        # y1 += 32 - h%32
        w, h = x1-x0, y1-y0
        im_crop = img[:, x0-16:x1+16, y0-16:y1+16]
        mask_crop = worker.mask[x0-16:x1+16, y0-16:y1+16]
        ch, w, h = im_crop.shape
        if worker.c.conv_resize:
            lx, ly = int(w/16), int(h/16)
        else:
            lx, ly = int(w/32) + 2, int(h/32) + 2
            
    target = im_crop.to(device)
    for ch in range(worker.c.n_phases):
        target[ch][mask_crop==1] = -1
    target = target.unsqueeze(0)
    ranges = [10000]
    lrs = [0.005, 0.001, 0.0005]
    kl_prefactors = [0, 0.001, 0.0001, 0.00001]
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    for k in kl_prefactors:
        for r in ranges:
            for lr in lrs:
                noise = [torch.nn.Parameter(torch.randn(1, worker.c.nz, lx, ly, requires_grad=True, device=device))]
                opt = torch.optim.Adam(params=noise, lr=lr)
                inpaints = []
                mses = []
                for i in range(int(r)):
                    raw = netG(noise[0])
                    loss = (raw - target)**2
                    loss[target==-1] = 0
                    loss = loss.sum() / ((target!=-1).sum()*loss.shape[1]*loss.shape[0])
                    mses.append(loss.item())
                    # Isaac to Steve - is this MSE an average over only the pixels that are valid? i.e. sum of loss / #pixels in the mask?
                    loss.backward()
                    opt.step()
                    if k>0:
                        loss_kl = k*kl_loss(noise[0], torch.randn_like(noise[0]))
                        loss_kl.backward()
                        opt.step()
                    with torch.no_grad():
                        noise[0] -= torch.tile(torch.mean(noise[0], dim=[1]), (1, worker.c.nz,1,1))
                        noise[0] /= torch.tile(torch.std(noise[0], dim=[1]), (1, worker.c.nz,1,1))
                if worker.c.image_type == 'n-phase':
                    raw = torch.argmax(raw[0], dim=0)[16:-16, 16:-16].detach().cpu()
                    out = torch.argmax(target.clone()[0], dim=0).detach().cpu()
                else:
                    raw = raw[0].permute(1,2,0)[16:-16, 16:-16].detach().cpu()
                    out = target[0].permute(1,2,0).detach().cpu()
                out[17:-15,16:-16] = raw
                print(k,r,lr,loss.item())
                df = border_analysis(out,c)
                plt.imshow(out, cmap='gray')
                plt.title(f'{k}, {r}, {lr}, {df.p[-1]}')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(f'analysis/{k}_{r}_{lr}.png')
                plt.close('all')

               

    # im_0 = inpaints[0]
    # im_10 = inpaints[1]
    # im_100 = inpaints[2]
    # im_1000 = inpaints[3]
    # im_10000 = inpaints[4]

    # fig = plt.figure()
    # gs = GridSpec(2,5)
    # ax_0 = fig.add_subplot(gs[0,0])
    # ax_10 = fig.add_subplot(gs[0,1])
    # ax_100 = fig.add_subplot(gs[0,2])
    # ax_1000 = fig.add_subplot(gs[0,3])
    # ax_10000 = fig.add_subplot(gs[0,4])
    # ax_mse = fig.add_subplot(gs[1,:])
    # ax_0.imshow(im_0, cmap='gray')
    # ax_0.set_axis_off()
    # ax_10.imshow(im_10, cmap='gray')
    # ax_10.set_axis_off()
    # ax_100.imshow(im_100, cmap='gray')
    # ax_100.set_axis_off()
    # ax_1000.imshow(im_1000, cmap='gray')
    # ax_1000.set_axis_off()
    # ax_10000.imshow(im_10000, cmap='gray')
    # ax_10000.set_axis_off()
    # ax_mse.plot(mses, color=global_colours[0])
    # ax_mse.set_xlabel('Iterations')
    # ax_mse.set_ylabel('MSE')
    # fig.tight_layout()
    # fig.savefig(f'analysis/{tag2}_zspan.png')
    # fig.savefig(f'analysis/{tag2}_zspan.eps', transparent=True)