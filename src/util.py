import numpy as np
import pandas as pd
import torch
from torch import autograd
import wandb
from dotenv import load_dotenv
import os
import subprocess
import shutil
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.patches import Rectangle

# check for existing models and folders
def check_existence(tag):
    """Checks if model exists, then asks for user input. Returns True for overwrite, False for load.

    :param tag: [description]
    :type tag: [type]
    :raises SystemExit: [description]
    :raises AssertionError: [description]
    :return: True for overwrite, False for load
    :rtype: [type]
    """
    root = f'runs/{tag}'
    check_D = os.path.exists(f'{root}/Disc.pt')
    check_G = os.path.exists(f'{root}/Gen.pt')
    if check_G or check_D:
        print(f'Models already exist for tag {tag}.')
        x = input("To overwrite existing model enter 'o', to load existing model enter 'l' or to cancel enter 'c'.\n")
        if x=='o':
            print("Overwriting")
            return True
        if x=='l':
            print("Loading previous model")
            return False
        elif x=='c':
            raise SystemExit
        else:
            raise AssertionError("Incorrect argument entered.")
    return True


# set-up util
def initialise_folders(tag, overwrite):
    """[summary]

    :param tag: [description]
    :type tag: [type]
    """
    if overwrite:
        try:
            os.mkdir(f'runs')
        except:
            pass
        try:
            os.mkdir(f'runs/{tag}')
        except:
            pass

def wandb_init(name, offline):
    """[summary]

    :param name: [description]
    :type name: [type]
    :param offline: [description]
    :type offline: [type]
    """
    if offline:
        mode = 'disabled'
    else:
        mode = None
    load_dotenv(os.path.join(os.getcwd(), '.env'))
    API_KEY = os.getenv('WANDB_API_KEY')
    ENTITY = os.getenv('WANDB_ENTITY')
    PROJECT = os.getenv('WANDB_PROJECT')
    if API_KEY is None or ENTITY is None or PROJECT is None:
        raise AssertionError('.env file arguments missing. Make sure WANDB_API_KEY, WANDB_ENTITY and WANDB_PROJECT are present.')
    print("Logging into W and B using API key {}".format(API_KEY))
    process = subprocess.run(["wandb", "login", API_KEY], capture_output=True)
    print("stderr:", process.stderr)

    
    print('initing')
    wandb.init(entity=ENTITY, name=name, project=PROJECT, mode=mode)

    wandb_config = {
        'active': True,
        'api_key': API_KEY,
        'entity': ENTITY,
        'project': PROJECT,
        # 'watch_called': False,
        'no_cuda': False,
        # 'seed': 42,
        'log_interval': 1000,

    }
    # wandb.watch_called = wandb_config['watch_called']
    wandb.config.no_cuda = wandb_config['no_cuda']
    # wandb.config.seed = wandb_config['seed']
    wandb.config.log_interval = wandb_config['log_interval']

def wandb_save_models(fn):
    """[summary]

    :param pth: [description]
    :type pth: [type]
    :param fn: [description]
    :type fn: filename
    """
    shutil.copy(fn, os.path.join(wandb.run.dir, fn))
    wandb.save(fn)

# training util
def preprocess(data_path, imtype, load=True):
    """[summary]

    :param imgs: [description]
    :type imgs: [type]
    :return: [description]
    :rtype: [type]
    """
    # img = tifffile.imread(data_path)
    img = plt.imread(data_path)
    if imtype == 'colour':
        img = img[:,:,:3]
        img = torch.tensor(img)
        if torch.max(img)>1:
            img = img /torch.max(img)
        return img.permute(2,0,1), 3
    else:
        if len(img.shape) > 2:
            img = img[...,0]
        if imtype == 'n-phase':
            phases = np.unique(img)
            if len(phases) > 10:
                raise AssertionError('Running in n-phase mode. Image exceeds max phases. Try running in colour or grayscale')
            x, y = img.shape
            img_oh = torch.zeros(len(phases), x, y)
            for i, ph in enumerate(phases):
                img_oh[i][img == ph] = 1
            return img_oh, len(phases)
        elif imtype == 'grayscale':
            img = np.expand_dims(img, 0)
            img = torch.tensor(img)
            if torch.max(img)>1:
                img = img /torch.max(img)
            return img, 1


def calculate_size_from_seed(seed, c):
    imsize = seed
    count = 0
    no_layers = len(c.gk)
    for (k, s, p) in zip(c.gk, c.gs, c.gp):
        if count<no_layers-2:
            imsize = (imsize-1)*s-2*p+k
        elif count==no_layers-2:
            imsize = ((imsize-k+2*p)/s+1).to(int)
            imsize = imsize*2+2
        else:
            imsize = ((imsize-k+2*p)/s+1).to(int)
        count +=1
    return imsize

def calculate_seed_from_size(imsize, c):
    count = 0
    no_layers = len(c.gk)
    for (k, s, p) in zip(c.gk, c.gs, c.gp):
        
        if count<no_layers-2:
            imsize = ((imsize-k+2*p)/s+1).to(int)
        elif count==no_layers-2:
            imsize = (imsize-1)*s-2*p+k
            imsize = ((imsize-2)/2).to(int)
        else:
            imsize = (imsize-1)*s-2*p+k
        count +=1
    return imsize

def make_mask(training_imgs, c):

    y1,y2,x1,x2 = c.mask_coords
    ydiff, xdiff = y2-y1, x2-x1

    # seed for size of inpainting region
    seed = calculate_seed_from_size(torch.tensor([xdiff, ydiff]).to(int), c)
    # add 2 seed to each side to make the MSE region, the total G region
    img_seed = seed+4
    G_out_size = calculate_size_from_seed(img_seed, c)
    mask_size = calculate_size_from_seed(seed, c)
    # THIS IS WHERE WE TELL D WHAT SIZE TO BE
    D_seed = img_seed
    x2, y2 = x1+mask_size[0].item(), y1+mask_size[1].item()
    xmid, ymid = (x2+x1)//2, (y2+y1)//2
    x1_bound, x2_bound, y1_bound, y2_bound = xmid-G_out_size[0].item()//2, xmid+G_out_size[0].item()//2, ymid-G_out_size[1].item()//2, ymid+G_out_size[1].item()//2
    unmasked = training_imgs[:,x1_bound:x2_bound, y1_bound:y2_bound].clone()
    training_imgs[:, x1:x2, y1:y2] = 0
    mask = training_imgs[:,x1_bound:x2_bound, y1_bound:y2_bound]
    mask_layer = torch.zeros_like(training_imgs[0]).unsqueeze(0)
    unmasked = torch.cat([unmasked, torch.zeros_like(unmasked[0]).unsqueeze(0)])
    mask_layer[:,x1:x2,y1:y2] = 1
    mask = torch.cat((mask, mask_layer[:,x1_bound:x2_bound, y1_bound:y2_bound]))

    # save coords to c
    c.img_seed_x, c.img_seed_y = (img_seed[0].item(), img_seed[1].item())
    c.mask_coords = (x1,x2,y1,y2)
    c.G_out_size = (G_out_size[0].item(), G_out_size[1].item())
    c.mask_size = (mask_size[0].item(), mask_size[1].item())
    c.D_seed_x = D_seed[0].item()
    c.D_seed_y = D_seed[1].item()
    return mask, unmasked, G_out_size, img_seed, c

def update_pixmap_rect(raw, img, c, save_path=None, border=False):
    # fig, ax = plt.subplots(211)
    # ax[0].imshow(img[0,0].detach().cpu().numpy())
    # ax[1].imshow(img[0,1].detach().cpu().numpy())
    # plt.imshow(img[0].cpu().permute(1,2,0).numpy())
    # plt.savefig('data/temp/raw.png')
    # plt.close()
    updated_pixmap = raw.clone().unsqueeze(0)
    x1, x2, y1, y2 = c.mask_coords
    lx, ly = c.mask_size
    x_1, x_2, y_1, y_2 = (img.shape[2]-lx)//2,(img.shape[2]+lx)//2, (img.shape[3]-ly)//2, (img.shape[3]+ly)//2
    updated_pixmap[:,:, x1:x2, y1:y2] = img[:,:,x_1:x_2, y_1:y_2]
    updated_pixmap = post_process(updated_pixmap, c).permute(0,2,3,1)
    if c.image_type=='grayscale':
        pm = updated_pixmap[0,...]
    else:
        pm = updated_pixmap[0].numpy()
    if save_path:
        fig, ax = plt.subplots()
        if c.image_type=='grayscale':
            ax.imshow(pm, cmap='gray')
            rect_col = '#CC2825'
        else:
            ax.imshow(pm)
            rect_col = "#CC2825"
            # rect_col = 'white'
            
        if border:
            rect = Rectangle((y1,x1),ly,lx,linewidth=1,ls='--', edgecolor=rect_col,facecolor='none')
            ax.add_patch(rect)
        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig('data/temp/temp_fig.png', transparent=True, pad_inches=0)
        plt.close()
        if c.image_type == 'grayscale':
            plt.imsave('data/temp/temp.png', np.concatenate([pm for i in range(3)], -1))
        else:
            plt.imsave('data/temp/temp.png', pm)
        return fig
    else:
        if c.image_type == 'grayscale':
            pm = np.concatenate([pm for i in range(3)], -1)
        plt.imsave('data/temp/temp.png', pm)
        return pm
    

def calc_gradient_penalty(netD, real_data, fake_data, batch_size, lx, ly, device, gp_lambda, nc):
    """[summary]

    :param netD: [description]
    :type netD: [type]
    :param real_data: [description]
    :type real_data: [type]
    :param fake_data: [description]
    :type fake_data: [type]
    :param batch_size: [description]
    :type batch_size: [type]
    :param l: [description]
    :type l: [type]
    :param device: [description]
    :type device: [type]
    :param gp_lambda: [description]
    :type gp_lambda: [type]
    :param nc: [description]
    :type nc: [type]
    :return: [description]
    :rtype: [type]
    """
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(
        real_data.nelement() / batch_size)).contiguous()
    alpha = alpha.view(batch_size, nc, lx, ly)
    alpha = alpha.to(device)

    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())
    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)
    disc_interpolates = netD(interpolates)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(
                                  disc_interpolates.size()).to(device),
                              create_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda
    return gradient_penalty

def batch_real_poly(img, l, bs, real_seeds):
    n_ph, _, _ = img.shape
    max_idx = len(real_seeds[0])
    idxs = torch.randint(max_idx, (bs,))
    data = torch.zeros((bs, n_ph, l, l))
    for i, idx in enumerate(idxs):
        x, y = real_seeds[0][idx], real_seeds[1][idx]
        data[i] = img[:, x:x+l, y:y+l]
    return data

def batch_real(img, lx, ly, bs, mask_coords):
    """[summary]
    :param training_imgs: [description]
    :type training_imgs: [type]
    :return: [description]
    :rtype: [type]
    """
    x1, x2, y1, y2 = mask_coords
    n_ph, x_max, y_max = img.shape
    data = torch.zeros((bs, n_ph, lx, ly))
    for i in range(bs):
        x, y = torch.randint(x_max - lx, (1,)), torch.randint(y_max - ly, (1,))
        while (x1<x+lx and x1>x-lx) and (y1<y+ly and y1>y-ly):
            x, y = torch.randint(x_max - lx, (1,)), torch.randint(y_max - ly, (1,))
        data[i] = img[:, x:x+lx, y:y+ly]
    return data

def pixel_wise_loss(fake_img, real_img, unmasked, mode='mse', device=None):
    mask = real_img.clone().permute(1,2,0)
    mask = (mask[...,-1]==0).unsqueeze(0)
    number_valid_pixels = mask.sum()
    mask = mask.repeat(fake_img.shape[0], fake_img.shape[1],1,1)
    fake_img = torch.where(mask==True, fake_img, torch.tensor(0).float().to(device))
    real_img = real_img.unsqueeze(0).repeat(fake_img.shape[0], 1 ,1, 1)[:,0:-1]
    real_img = torch.where(mask==True, real_img, torch.tensor(0).float().to(device))

    if mode=='mse':
        loss = torch.nn.MSELoss(reduction='sum')(fake_img, real_img)/(number_valid_pixels*fake_img.shape[0]*fake_img.shape[1])
    elif mode=='ce':
        loss = -(real_img*torch.log(fake_img) + (1-real_img)*torch.log(1-fake_img)).nanmean()

    return loss

# Evaluation util
def post_process(img, c):
    """Turns a n phase image (bs, n, imsize, imsize) into a plottable euler image (bs, 3, imsize, imsize, imsize)

    :param img: a tensor of the n phase img
    :type img: torch.Tensor
    :return:
    :rtype:
    """
    img = img.detach().cpu()
    if c.image_type=='n-phase':
        phases = np.arange(c.n_phases)
        color = iter(cm.get_cmap(c.cm)(np.linspace(0, 1, c.n_phases)))
        # color = iter([[0,0,0],[0.5,0.5,0.5], [1,1,1]])
        img = torch.argmax(img, dim=1)
        if len(phases) > 10:
            raise AssertionError('Image not one hot encoded.')
        bs, x, y = img.shape
        out = torch.zeros((bs, 3, x, y))
        for b in range(bs):
            for i, ph in enumerate(phases):
                col = next(color)
                col = torch.tile(torch.Tensor(col[0:3]).unsqueeze(1).unsqueeze(1), (x,y))
                out[b] = torch.where((img[b] == ph), col, out[b])
        out = out
    else:
        out = img
    return out

def crop(fake_data, l, miniD=False, l_mini=16, offset=8):
    w = fake_data.shape[2]
    h = fake_data.shape[3]
    x1,x2 = (w-l)//2,(w+l)//2
    y1,y2 = (h-l)//2,(h+l)//2
    
    out = fake_data[:,:,x1:x2, y1:y2]
    return out

def init_noise(batch_size, nz, c, device):
    noise = torch.randn(1, nz, c.seed_x, c.seed_y, device=device)
    noise = torch.tile(noise, (batch_size, 1, 1 ,1))
    noise.requires_grad = True
    return noise

def make_noise(noise, device, mask_noise=False, delta=[1,1]):
    # zeros in mask are fixed, ones are random
    mask = torch.zeros_like(noise).to(device)
    _, _, x, y = mask.shape
    dx = delta[0]//2
    dy = delta[1]//2
    # 
    if mask_noise:
        if dx>0 and dy>0:
            mask[:,:,x//2-dx:x//2+dx,y//2-dy:y//2+dy] = 1
        elif dx==0:
            mask[:,:,x//2,y//2-dy:y//2+dy] = 1
        elif dy==0:
            mask[:,:,x//2-dx:x//2+dx,y//2] = 1
        rand = torch.randn_like(noise).to(device)*mask
        noise = noise*(mask==0)+rand
    else:
        noise = torch.randn_like(noise).to(device)
    # plt.imshow(mask[0,0].detach().cpu().numpy())
    # plt.savefig('noise.png')
    return noise