import numpy as np
import torch
from torch import autograd
import wandb
from dotenv import load_dotenv
import os
import subprocess
import shutil
import matplotlib.pyplot as plt
from torch import nn
import tifffile

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
def preprocess(data_path):
    """[summary]

    :param imgs: [description]
    :type imgs: [type]
    :return: [description]
    :rtype: [type]
    """
    # img = tifffile.imread(data_path)
    img = plt.imread(data_path)[...,0]
    phases = np.unique(img)
    if len(phases) > 10:
        raise AssertionError('Image not one hot encoded.')
    # x, y, z = img.shape
    x, y = img.shape
    # img_oh = torch.zeros(len(phases), x, y, z)
    img_oh = torch.zeros(len(phases), x, y)
    for i, ph in enumerate(phases):
        img_oh[i][img == ph] = 1
    return img_oh, len(phases)

def make_mask(training_imgs, mask_coords):
    y1,y2,x1,x2 = mask_coords
    ydiff, xdiff = y2-y1, x2-x1
    maxdiff = np.max([ydiff, xdiff])
    maxdiff=64
    x2, y2 = x1+maxdiff, y1+maxdiff
    x1_bound, x2_bound, y1_bound, y2_bound = x1-maxdiff//2, x2+maxdiff//2, y1-maxdiff//2, y2+maxdiff//2
    
    unmasked = training_imgs[:,x1_bound:x2_bound, y1_bound:y2_bound].clone()
    training_imgs[:, x1:x2, y1:y2] = 0
    mask = training_imgs[:,x1_bound:x2_bound, y1_bound:y2_bound]
    mask_layer = torch.zeros_like(training_imgs[0]).unsqueeze(0)
    unmasked = torch.cat([unmasked, torch.zeros_like(unmasked[0]).unsqueeze(0)])
    mask_layer[:,x1:x2,y1:y2] = 1
    mask = torch.cat((mask, mask_layer[:,x1_bound:x2_bound, y1_bound:y2_bound]))

    plt.imsave('data/mask.png',mask.permute(1,2,0).numpy())
    plt.imsave('data/unmasked.png',unmasked.permute(1,2,0).numpy())
    return mask, unmasked

def calc_gradient_penalty(netD, real_data, fake_data, batch_size, l, device, gp_lambda, nc):
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
    alpha = alpha.view(batch_size, nc, l, l)
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

def batch_real(img, l, bs):
    """[summary]
    :param training_imgs: [description]
    :type training_imgs: [type]
    :return: [description]
    :rtype: [type]
    """
    n_ph, x_max, y_max = img.shape
    data = torch.zeros((bs, n_ph, l, l))
    for i in range(bs):
        x, y = torch.randint(x_max - l, (1,)), torch.randint(y_max - l, (1,))
        data[i] = img[:, x:x+l, y:y+l]
    return data

def pixel_wise_loss(fake_img, real_img, coeff=1, device=None):
    mask = real_img.clone().permute(1,2,0)
    mask = (mask[...,-1]==0).unsqueeze(0)
    mask = mask.repeat(fake_img.shape[0], fake_img.shape[1],1,1)
    fake_img = torch.where(mask==True, fake_img, torch.tensor(0).float().to(device))
    real_img = real_img.unsqueeze(0).repeat(fake_img.shape[0], 1 ,1, 1)[:,0:3]
    real_img = torch.where(mask==True, real_img, torch.tensor(0).float().to(device))
    return torch.nn.MSELoss(reduction='none')(fake_img, real_img)*coeff

# Evaluation util
def post_process(img, phases=[0,1,2]):
    """Turns a n phase image (bs, n, imsize, imsize) into a plottable euler image (bs, 3, imsize, imsize, imsize)

    :param img: a tensor of the n phase img
    :type img: torch.Tensor
    :return:
    :rtype:
    """
    img = img.detach().cpu()
    img = torch.argmax(img, dim=1).numpy()
    # phases = np.unique(img)
    if len(phases) > 10:
        raise AssertionError('Image not one hot encoded.')
    bs, x, y = img.shape
    img_oh = torch.zeros(bs, len(phases), x, y)
    for b in range(bs):
        for i, ph in enumerate(phases):
            img_oh[b,i][img[b] == ph] = 1
    return img_oh

def generate(c, netG, skeleton):
    """Generate an instance from generator, save to .tif

    :param c: Config object class
    :type c: Config
    :param netG: Generator instance
    :type netG: Generator
    :return: Post-processed generated instance
    :rtype: torch.Tensor
    """
    tag, ngpu, nz, lf, pth = c.tag, c.ngpu, c.nz, c.lf, c.path


    out_pth = f"runs/{tag}/out.tif"
    if torch.cuda.device_count() > 1 and c.ngpu > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
    device = torch.device("cuda:0" if(
            torch.cuda.is_available() and ngpu > 0) else "cpu")
    if (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu))).to(device)
    netG.load_state_dict(torch.load(f"{pth}/Gen.pt"))
    netG.eval()
    noise = torch.randn(1, nz, lf, lf)
    raw = netG(noise, skeleton)
    gb = post_process(raw)
    tif = np.array(gb[0].permute(1,2,0), dtype=np.uint8)
    tifffile.imwrite(out_pth, tif, imagej=True)
    return tif

def progress(i, iters, n, num_epochs, timed):
    """[summary]

    :param i: [description]
    :type i: [type]
    :param iters: [description]
    :type iters: [type]
    :param n: [description]
    :type n: [type]
    :param num_epochs: [description]
    :type num_epochs: [type]
    :param timed: [description]
    :type timed: [type]
    """
    progress = 'iteration {} of {}, epoch {} of {}'.format(
        i, iters, n, num_epochs)
    print(f"Progress: {progress}, Time per iter: {timed}")

def plot_img(img, iter, epoch, path, offline=True):
    """[summary]

    :param img: [description]
    :type img: [type]
    :param slcs: [description], defaults to 4
    :type slcs: int, optional
    """
    img = post_process(img)
    if not offline:
        wandb.log({"slices": [wandb.Image(i[:, 31]) for i in img]})
    else:
        plt.imsave(f'{path}/test.png', img[0][:, 31])


def plot_examples(img, mask, unmasked, mse, offline=True):
    """[summary]

    :param img: [description]
    :type img: [type]
    :param slcs: [description], defaults to 4
    :type slcs: int, optional
    """
    if not offline:
        fig, ax = plt.subplots(2,3)
        fig.suptitle('inpainting')
        ax[0,0].set_title('Original')
        ax[0,1].set_title('Mask')
        ax[0,2].set_title('Original-masked')
        ax[1,0].set_title('G output')
        ax[1,1].set_title('Inpainted')
        ax[1,2].set_title('MSE loss')

        # ax[0,0].imshow(post_process(unmasked.unsqueeze(0))[0].permute(1,2,3,0).cpu()[32])
        # ax[0,1].imshow(mask[-1, 32].cpu())
        # ax[0,2].imshow(post_process(mask.unsqueeze(0))[0,0:3].permute(1,2,3,0).cpu()[32])
        # ax[1,0].imshow(post_process(img)[0].permute(1,2,3,0).cpu()[32])
        # ax[1,1].imshow(post_process(inpaint(img, unmasked))[0].permute(1,2,3,0).cpu()[32])
        # ax[1,2].imshow(mse[0].permute(1,2,3,0).cpu()[32])
        ax[0,0].imshow(post_process(unmasked.unsqueeze(0))[0].permute(1,2,0).cpu())
        ax[0,1].imshow(mask[-1, 32].cpu())
        ax[0,2].imshow(post_process(mask.unsqueeze(0))[0,0:3].permute(1,2,0).cpu())
        ax[1,0].imshow(post_process(img)[0].permute(1,2,0).cpu())
        ax[1,1].imshow(post_process(inpaint(img, unmasked))[0].permute(1,2,0).cpu())
        ax[1,2].imshow(mse[0].permute(1,2,0).cpu())
        fig.tight_layout()
        wandb.log({"examples": wandb.Image(fig)})
        plt.close()
    
def inpaint(fake_data, unmasked):
    l = fake_data.shape[2]
    unmasked = unmasked.unsqueeze(0).repeat(fake_data.shape[0],1,1,1)
    out = unmasked.clone()
    out[:,:, l//4:3*l//4,l//4:3*l//4,4] = fake_data[:,:,l//4:3*l//4,l//4:3*l//4]
    return out

def crop(fake_data, l):
    w = fake_data.shape[2]
    return fake_data[:,:,w//2-l//2:w//2+l//2,w//2-l//2:w//2+l//2]

def make_noise(bs, nz, lz, device):
    noise = torch.ones(bs, nz, lz, lz, device=device)
    noise[:,:,lz//2,lz//2,] = torch.randn(bs,nz)
    return noise