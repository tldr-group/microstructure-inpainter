import argparse
import json
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import torch
from config import Config, ConfigPoly
from src import networks, util
from src.train_poly import PolyWorker
from src.train_rect import RectWorker
from scipy.stats import ks_2samp
from cycler import cycler
import torch.nn as nn

global_colours = ['#253659', '#04BF9D', '#F27457', '#C276C1', '#EBE57E']
prop_cycler = cycler(color=global_colours)

font = {'size': 8}
matplotlib.rc('font', **font)

def main(tag1, tag2, generate=False, metric_compare=False, load=False, z_span=False, borders=False, seed_prop=False):
    
    if generate or metric_compare or z_span or seed_prop or (borders and tag1 != 'empty'):

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
        delta_x, delta_y = (c.G_out_size[0]-lx) //2, (c.G_out_size[1]-ly) //2
        img, n_phases = util.preprocess(path, image_type)
        img = (img/img.max())

        if generate:
            # save ground truth
            fig = plt.figure()
            # fig.suptitle(f"{tag1} - {tag2}")
            if tag1 =='empty' or tag2 == 'empty':
                gs = GridSpec(1,3)
                if tag1 == 'empty':
                    tag2_index = 0
                else:
                    tag1_index = 0
            else:
                gs = GridSpec(2,3)
                tag1_index = 0
                tag2_index = 1
            axGT = fig.add_subplot(gs[:,0])
            axGT.set_title("Ground truth")
            # ax.imshow(np.stack([img for i in range(3)], -1))
            if image_type=='grayscale':
                gt = util.post_process(img[:,y1-delta_y:y2+delta_y, x1-delta_x:x2+delta_x].unsqueeze(0),c)[0].permute(1,2,0)
                axGT.imshow(gt, cmap='gray')
                rect_col = '#CC2825'
            else:
                gt = util.post_process(img[:,y1-delta_y:y2+delta_y, x1-delta_x:x2+delta_x].unsqueeze(0),c)[0].permute(1,2,0)
                axGT.imshow(gt)
                rect_col='#CC2825'
            rect = Rectangle((delta_x, delta_y),lx,ly,linewidth=1,ls='--', edgecolor=rect_col,facecolor='none')
            axGT.add_patch(rect)
            axGT.set_axis_off()

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
            real_vfs = ()
            for p in range(n_phases):
                    real_vfs = real_vfs + (img[p].mean().item(),)
                    
                
            

        # RECT
        if tag1 != 'empty':
            c = Config(tag1)
            c.load()
            c.mask_coords = (x1, x2, y1, y2)
            overwrite = False
            training_imgs, nc = util.preprocess(c.data_path, c.image_type)
            mask, unmasked, img_size, seed, c = util.make_mask(training_imgs, c)
            netD, netG = networks.make_nets(c, overwrite)
            worker = RectWorker(c, netG, netD, training_imgs, nc, mask, unmasked)
            worker.verbose = True
            if seed_prop:
                netG = worker.netG()
                s = 20
                plt.close('all')
                noise = torch.randn(1, 100, s, s)
                print(netG(torch.randn(1, 100, 4, 4)).shape)
                baseline = netG(noise)[0, 0]
                print(baseline.shape)
                for c in range(10):
                    st = s//2-c
                    noise[:,:,st:-st,st:-st] = torch.randn(1, 100, c*2, c*2)
                    out = netG(noise)[0, 0]
                    mse = (out - baseline)**2
                    mse = torch.sum(mse, dim=1)
                    print((mse>0.005).sum(), c*2)
                    plt.plot(mse.detach(), label=c*2)
                plt.legend()
                plt.grid()
                plt.xticks(np.arange(0, len(mse), 2))
                plt.savefig('analysis/seed.png')
            if generate:
                axRectAll = fig.add_subplot(gs[0,1:])
                axRectAll.annotate('G opt.', xy=(1.15, 0.5), xytext=(0, 0),
                        xycoords='axes fraction', textcoords='offset points',
                        size='large', ha='center', va='baseline')
                axRectAll.set_axis_off()
                for i in range(2):
                    axRECT = fig.add_subplot(gs[tag1_index,i+1])
                    img = worker.generate(save_path=f'analysis/{tag1}_{i}', border=True)
                    p = util.post_process(img,c)[0].permute(1,2,0).numpy()
                    img = inpaint(gt, util.post_process(img,c)[0].permute(1,2,0))
                    axRECT.imshow(img, cmap='gray')
                    axRECT.set_axis_off()
                    axRECT.set_title(f'Seed {i+1}')
                fig.tight_layout()
                fig.savefig(f'analysis/{tag1}_{tag2}.png', transparent=False)
                fig.savefig(f'analysis/{tag1}_{tag2}_transparent.png', transparent=True)
                fig.savefig(f'analysis/{tag1}_{tag2}.eps', transparent=True)

            if metric_compare and not load:
                rect_list_rand = []
                rect_list_fixed = []
                for i in range(samples):
                    for ls, delta in zip([rect_list_rand, rect_list_fixed],['rand',None]):
                        im = worker.generate(delta=delta)[0].cpu()
                        im = (torch.argmax(im, dim=0))
                        x,y = im.shape
                        im = im[16:-16,16:-16,]
                        ph = np.unique(im)
                        vfs = ()
                        for p in ph:
                            l = im==p
                            vfs = vfs + (l.float().mean().item(),)
                        ls.append(vfs)

            if borders:
                rect_im = worker.generate()
        if tag2 != 'empty':
            # POLY

            c = ConfigPoly(tag2)
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
            netD, netG = networks.make_nets(c, overwrite)
            worker = PolyWorker(c, netG, netD, real_seeds, mask, poly_rects, c.frames, overwrite)
            worker.verbose = True
            if z_span:
                plot_z_span(worker, tag2)
                

            if generate:
                axPolyAll = fig.add_subplot(gs[1,1:])
                axPolyAll.annotate('z opt.', xy=(1.15, 0.5), xytext=(0, 0),
                        xycoords='axes fraction', textcoords='offset points',
                        size='large', ha='center', va='baseline')
                axPolyAll.set_axis_off()
                for i in range(2):
                    axPOLY = fig.add_subplot(gs[tag2_index,i+1])
                    img = worker.generate(save_path=f'analysis/{tag2}_{i}', border=True, opt_iters=10000)
                    img = inpaint(gt, util.post_process(img,c)[0].permute(1,2,0).detach())
                    axPOLY.imshow(img, cmap='gray')
                    axPOLY.set_axis_off()
                    # axPOLY.set_title(f'Seed {i+1}')
                fig.tight_layout()
                plt.subplots_adjust(hspace=0)
                fig.savefig(f'analysis/{tag1}_{tag2}.png', transparent=False)
                fig.savefig(f'analysis/{tag1}_{tag2}_transparent.png', transparent=True)
                fig.savefig(f'analysis/{tag1}_{tag2}.eps', transparent=True)

            if metric_compare:
                if not load:
                    poly_list_unopt = []
                    poly_list_opt = []
                    for i in range(samples):
                        for opt_iters in [0,10000]:
                            im = worker.generate(opt_iters=opt_iters)[0].detach().cpu()
                            im = (torch.argmax(im, dim=0))
                            x,y = im.shape
                            im = im[(x-lx)//2:(x+lx)//2,(y-ly)//2:(y+ly)//2,]
                            ph = np.unique(im)
                            vfs = ()
                            for p in ph:
                                i = im==p
                                vfs = vfs + (i.float().mean().item(),)
                            if opt_iters>0:
                                poly_list_opt.append(vfs)
                            else:
                                poly_list_unopt.append(vfs)


                    vfs_dict = {}
                    type_list = ['Real', 'G rand', 'G fixed', 'z opt', 'z unopt']
                    labels = []
                    data = []
                    for i, ls in enumerate([real_list, rect_list_rand, rect_list_fixed, poly_list_opt, poly_list_unopt]):
                        label = type_list[i]
                        vfs_dict[label] = {}
                        if label =='Real':
                            vfs_dict['Real']['Total'] = {}
                        for j, ph in enumerate(['Phase 1', "Phase 2", "Phase 3"]):
                            vfs_dict[label][ph] = [a[j] for a in ls]
                            labels.append(label+' '+ph)
                            data.append([a[j] for a in ls])
                            if ls == real_list:
                                vfs_dict['Real']['Total'][ph] = real_vfs[j]
                    with open(f'analysis/vfs_{tag1}_{tag2}_analysis.json', 'w') as fp:
                                json.dump(vfs_dict, fp)
                if load:
                    data = []
                    labels = []
                    colours = []
                plot_vfs(tag1, tag2, data=data, labels=labels, load=load)
            
            if borders:
                poly_im = worker.generate()
                poly_im_unopt = worker.generate(opt_iters=0)
                # plot_border_analysis_specfic(tag1, tag2, c,rect_im.detach().cpu(),poly_im.detach().cpu())
                border_contiguity_analysis(tag1, tag2, c, rect_im.detach().cpu(), poly_im.detach().cpu(), poly_im_unopt.detach().cpu())

def inpaint(gt, im):
    gt[16:-16,16:-16] = im[16:-16, 16:-16]
    return gt

def plot_z_span(worker, tag2):
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
        lx, ly = util.calculate_seed_from_size(torch.tensor([w,h]), worker.c)
        w, h = util.calculate_size_from_seed(torch.tensor([lx,ly]), worker.c)
        x1 = x0 + w
        y1 = y0 + h
        im_crop = img[:, x0-16:x1+16, y0-16:y1+16]
        mask_crop = worker.mask[x0-16:x1+16, y0-16:y1+16]
    target = im_crop.to(device)
    for ch in range(worker.c.n_phases):
        target[ch][mask_crop==1] = -1
    target = target.unsqueeze(0)
    noise = [torch.nn.Parameter(torch.randn(1, worker.c.nz, lx+4, ly+4, requires_grad=True, device=device))]
    opt = torch.optim.Adam(params=noise, lr=worker.c.opt_lr)
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    inpaints = []
    mses = []
    for i in range(int(10000)):
        raw = netG(noise[0])
        loss = (raw - target)**2
        loss[target==-1] = 0
        loss_copy = loss.clone()
        loss = loss.mean()
        loss.backward()
        opt.step()
        mses.append(loss_copy.mean(dim=(1,2,3)).min().detach().cpu())
        # Isaac to Steve - is this MSE an average over only the pixels that are valid? i.e. sum of loss / #pixels in the mask?
        loss_kl = worker.c.opt_kl_coeff*kl_loss(noise[0], torch.randn_like(noise[0]))
        loss_kl.backward()
        opt.step()
        with torch.no_grad():
            noise[0] -= torch.tile(torch.mean(noise[0], dim=[1]), (1, worker.c.nz,1,1))
            noise[0] /= torch.tile(torch.std(noise[0], dim=[1]), (1, worker.c.nz,1,1))
        if i in saves_list:
            print(i)
            if worker.c.image_type == 'n-phase':
                raw = torch.argmax(raw[0], dim=0)[16:-16, 16:-16].detach().cpu()
                out = torch.argmax(target.clone()[0], dim=0).detach().cpu()
            else:
                raw = raw[0].permute(1,2,0)[16:-16, 16:-16].detach().cpu()
                out = target[0].permute(1,2,0).detach().cpu()
            out[17:-15,16:-16] = raw
            inpaints.append(out)

    im_0 = inpaints[0]
    im_10 = inpaints[1]
    im_100 = inpaints[2]
    im_1000 = inpaints[3]
    im_10000 = inpaints[4]

    fig = plt.figure()
    gs = GridSpec(2,5)
    ax_0 = fig.add_subplot(gs[0,0])
    ax_10 = fig.add_subplot(gs[0,1])
    ax_100 = fig.add_subplot(gs[0,2])
    ax_1000 = fig.add_subplot(gs[0,3])
    ax_10000 = fig.add_subplot(gs[0,4])
    ax_mse = fig.add_subplot(gs[1,:])
    ax_0.imshow(im_0, cmap='gray')
    ax_0.set_axis_off()
    ax_10.imshow(im_10, cmap='gray')
    ax_10.set_axis_off()
    ax_100.imshow(im_100, cmap='gray')
    ax_100.set_axis_off()
    ax_1000.imshow(im_1000, cmap='gray')
    ax_1000.set_axis_off()
    ax_10000.imshow(im_10000, cmap='gray')
    ax_10000.set_axis_off()
    ax_mse.plot(mses, color=global_colours[0])
    for x,y in zip(saves_list, [mses[i] for i in saves_list]):
        ax_mse.plot(x,y, 'x', color=global_colours[2])
    ax_mse.set_xlabel('Iterations')
    ax_mse.set_ylabel('MSE')
    fig.tight_layout()
    fig.savefig(f'analysis/{tag2}_zspan.png')
    fig.savefig(f'analysis/{tag2}_zspan.eps', transparent=True)


def plot_vfs(tag1, tag2, data=None, labels=None, load=True):
    with open(f'analysis/vfs_{tag1}_{tag2}_analysis.json', 'r') as fp:
        loaded = json.load(fp)
    colours = []
    labels = []
    colours_list = global_colours
    data_G_rand = []
    data_G_fixed = []
    data_r = []
    data_r_total = []
    data_z_unopt = []
    data_z_opt = []
    phases = []
    key_list = loaded[list(loaded.keys())[1]].keys()

    # KS test on each phase
    ks_test_res = {}
    gt = loaded['Real']
    for m in loaded.keys():
        ks_test_res[m] = {}
        for p in loaded[m].keys():
            if p!='Total':
                p_val = ks_2samp(gt[p], loaded[m][p])[-1]
                ks_test_res[m][p] = f'{p_val:.2g}'
    with open(f'analysis/vf_ks_{tag1}_{tag2}.json', 'w') as f:
        json.dump(ks_test_res, f)
    ks_df = pd.DataFrame(ks_test_res)
    with open(f"analysis/vf_ks_{tag1}_{tag2}.tex", "w") as f:
        l = ks_df.to_latex(buf=f, bold_rows=True,label='vf_ks', index=False)


    for j, k2 in enumerate(key_list):
        labels.append('Real')
        data_r.append(loaded['Real'][k2])
        data_r_total.append(loaded['Real']['Total'][k2])

        labels.append('G opt')
        data_G_fixed.append(loaded['G fixed'][k2])
        data_G_rand.append(loaded['G rand'][k2])

        data_z_unopt.append(loaded['z unopt'][k2])
        data_z_opt.append(loaded['z opt'][k2])
        labels.append('z opt')

        colours.append(colours_list[j])
        phases.append(k2)
            
    fig, ax = plt.subplots()
    parts_r = ax.violinplot(data_r, showextrema=False, showmeans=True, positions=[1,4,7])
    parts_G_fixed = ax.violinplot(data_G_fixed, showextrema=False, showmeans=True, positions=[2,5,8])
    parts_G_rand = ax.violinplot(data_G_rand, showextrema=False, showmeans=True, positions=[2,5,8])
    parts_z_unopt = ax.violinplot(data_z_unopt, showextrema=False, showmeans=True, positions=[3,6,9])
    parts_z_opt = ax.violinplot(data_z_opt, showextrema=False, showmeans=True, positions=[3,6,9])
    
    for i, pc in enumerate(parts_r['bodies']):
        pc.set_facecolor(colours[i])
        pc.set_alpha(1)


    # Make G opts half violins
    for i, b in enumerate(parts_G_rand['bodies']):
        # get the center
        m = np.mean(b.get_paths()[0].vertices[:, 0])
        # modify the paths to not go further right than the center
        b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
        b.set_facecolor(colours[i])
        b.set_edgecolor('white')
        b.set_alpha(1)

    for j in parts_G_rand['cmeans'].get_paths():
        m = np.mean(j.vertices[:,0])
        j.vertices[:,0] = np.clip(j.vertices[:,0], -np.inf, m)

    for i, b in enumerate(parts_G_fixed['bodies']):
        # get the center
        m = np.mean(b.get_paths()[0].vertices[:, 0])
        # modify the paths to not go further right than the center
        b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
        b.set_facecolor(colours[i])
        b.set_edgecolor('white')
        b.set_alpha(1)

    for j in parts_G_fixed['cmeans'].get_paths():
        m = np.mean(j.vertices[:,0])
        j.vertices[:,0] = np.clip(j.vertices[:,0], m, np.inf)

        # Make z opts half violins
    for i, b in enumerate(parts_z_unopt['bodies']):
        # get the center
        m = np.mean(b.get_paths()[0].vertices[:, 0])
        # modify the paths to not go further right than the center
        b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
        b.set_facecolor(colours[i])
        b.set_edgecolor('white')
        b.set_alpha(1)
        
    for j in parts_z_unopt['cmeans'].get_paths():
        m = np.mean(j.vertices[:,0])
        j.vertices[:,0] = np.clip(j.vertices[:,0], -np.inf, m)

    for i, b in enumerate(parts_z_opt['bodies']):
        # get the center
        m = np.mean(b.get_paths()[0].vertices[:, 0])
        # modify the paths to not go further right than the center
        b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
        b.set_facecolor(colours[i])
        b.set_edgecolor('white')
        b.set_alpha(1)

    for j in parts_z_opt['cmeans'].get_paths():
        m = np.mean(j.vertices[:,0])
        j.vertices[:,0] = np.clip(j.vertices[:,0], m, np.inf)
    
    for k in [parts_r, parts_G_fixed, parts_G_rand, parts_z_opt, parts_z_unopt]:
        k['cmeans'].set_edgecolor('white')

    patches = []
    phs = []
    phs = ['pore', 'metal', 'ceramic']

    for p, c in zip(pd.unique(phases),pd.unique(colours)):
        patches.append(mpatches.Patch(color=c, alpha=1))
        # phs.append(p)
    ax.legend(patches, phs)
    ax.set_xticks(np.arange(1,len(labels)+1))
    ax.set_xticklabels(labels)

    ax.set_ylabel('Volume fraction')
    plt.savefig(f'analysis/{tag1}_{tag2}_vf_violinplot.eps', transparent=True)
    plt.savefig(f'analysis/{tag1}_{tag2}_vf_violinplot_transparent.png', transparent=True)
    plt.savefig(f'analysis/{tag1}_{tag2}_vf_violinplot.png')


def border_contiguity_analysis(tag1, tag2, c, rect_im, poly_im, poly_im_unopt, compare_all=True):
    print(f'Analysing border matching for {tag1} and {tag2}')
    size = 16
    x1=c.mask_coords[2]-size
    y1=c.mask_coords[0]-size
    x2=c.mask_coords[3]+size
    y2=c.mask_coords[1]+size
    l = x2-x1-2*size
    gt, _ = util.preprocess(c.data_path, c.image_type)
    gt = gt.permute(1,2,0)[x1:x2,y1:y2].numpy()
    inpaint_zeros = gt.copy()
    inpaint_zeros[size:-size,size:-size] = 0

    inpaint_noise = gt.copy()
    inpaint_noise[size:-size,size:-size] = np.random.uniform(size=(l,l,1))

    # rect_im = util.post_process(rect_im, c)
    # poly_im = util.post_process(poly_im, c)
    rect_im = torch.argmax(rect_im, dim=1)
    poly_im = torch.argmax(poly_im, dim=1)
    rect_im = torch.nn.functional.one_hot(rect_im).permute(0, 3, 1, 2)
    poly_im = torch.nn.functional.one_hot(poly_im).permute(0, 3, 1, 2)

    g_opt_im = gt.copy()
    z_opt_im = gt.copy()
    g_opt_im[size:-size, size:-size]=rect_im[0].permute(1,2,0)[size:-size,size:-size]
    z_opt_im[size:-size, size:-size]=poly_im[0].permute(1,2,0)[size:-size,size:-size]
    diffs = []
    data_list = [gt, g_opt_im, z_opt_im]
    name_list = ['Ground truth', 'G opt.', 'z opt.']
    axes_list = [[0],[1],[2]]

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
    # plot comparison of GT and two methods
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
        plt.tight_layout()
        fig.savefig(f'analysis/{tag1}_{tag2}_border_contiguity_analysis_transparent.png', transparent=True)
        fig.savefig(f'analysis/{tag1}_{tag2}_border_contiguity_analysis.png')
        fig.savefig(f'analysis/{tag1}_{tag2}_border_contiguity_analysis.eps', transparent=True)

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

    df = pd.DataFrame({'Name': name_list,
                        # 'Mean MSE': [f'{np.mean(t):.2g} ± {np.std(t):.2g}' for t in diffs],
                        'KS value': [f"{ks_2samp(t, gt_diff)[0]:.2g}" for t in diffs],
                        'p': [f"{ks_2samp(t, gt_diff)[-1]:.2g}" for t in diffs]})

    # df.index = df['Name']
    # df.drop("Name", axis=1, inplace=True)
    print(df.head())
    with open(f"analysis/{tag1}_{tag2}_border_analysis.tex", "w") as f:
        l = df.to_latex(buf=f, bold_rows=True,label='contiguity_validation', index=False)
    
    # plot comparison of GT, noise, zeros, bad and good inpaint
    if compare_all:
        diffs = []
        z_unopt_im = gt.copy()
        z_unopt_im[size:-size, size:-size]=poly_im_unopt[0].permute(1,2,0)[size:-size,size:-size]
        data_list = [gt, inpaint_zeros, inpaint_noise, z_unopt_im]
        name_list = ['Ground truth', 'Zeros', 'Noise', 'Random seed']
        axes_list = [[0,0],[0,1],[1,0],[1,1]]

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
        fig = plt.figure()
        gs = GridSpec(2,2)
        for d, a, n in zip(data_list, axes_list, name_list):
            if len(a)>1:
                x,y = a
                ax = fig.add_subplot(gs[x,y])
            else:
                ax = fig.add_subplot(gs[:,a[0]])
            ax.imshow(d, cmap='gray')
            ax.set_title(n)
            ax.set_axis_off()
            plt.tight_layout()
            fig.savefig('analysis/border_contiguity_analysis.png')
            fig.savefig('analysis/border_contiguity_analysis_transparent.png', transparent=True)
            fig.savefig('analysis/border_contiguity_analysis.eps', transparent=True)
        
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

        df = pd.DataFrame({'Name': ['Ground truth', 'Zeros', 'Noise', 'Random seed'],
                            # 'Volume fraction': [f'{np.mean(t):.2g} ± {np.std(t):.2g}' for t in diffs],
                            'KS value': [f"{ks_2samp(t, gt_diff)[0]:.2g}" for t in diffs],
                            'p': [f"{ks_2samp(t, gt_diff)[-1]:.2g}" for t in diffs]})

        # df.index = df['Name']
        # df.drop("Name", axis=1, inplace=True)
        print(df.head())
        with open("analysis/border_analysis.tex", "w") as f:
            l = df.to_latex(buf=f, bold_rows=True,label='contiguity_validation', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--generate", default=False)
    parser.add_argument("-mse", default=False)
    parser.add_argument("-wass", default=False)
    parser.add_argument("-metric", default=False)
    parser.add_argument("-zspan", default=False)
    parser.add_argument("-borders", default=False)
    parser.add_argument("-seed", default=False)
    parser.add_argument("-t1", "--tag1", default='empty')
    parser.add_argument("-t2", "--tag2", default='empty')
    


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
    if args.zspan=='t':
        z_span=True
    else:
        z_span=False
    if args.borders=='t':
        borders=True
    else:
        borders=False
    if args.seed=='t':
        seed_prop=True
    else:
        seed_prop=False
    if args.metric=='load':
        metric=True
        load=True
    elif args.metric=='gen':
        metric=True
        load=False
    else:
        metric=False
        load=False

    main(args.tag1, args.tag2, generate=generate, metric_compare=metric, load=load, z_span=z_span, borders=borders, seed_prop=seed_prop)
    # plot_vfs(tag='case1_rect', load=True)
    # main('train', True, 'test')

