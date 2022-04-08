import os
from bayes_opt import BayesianOptimization
import torch
import torch.nn as nn
from config import Config
import src.networks as networks
from src.util import inpaint, pixel_wise_loss, load_mask, post_process
import matplotlib.pyplot as plt
import imageio
import re

c = Config('no-matching')
nz, lf, pth, ngpu = c.nz, c.lf, c.path, c.ngpu

iters = 9

netD, netG = networks.make_nets(c, False)
netG = netG()
if torch.cuda.device_count() > 1 and c.ngpu > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
device = torch.device(c.device_name if(
        torch.cuda.is_available() and ngpu > 0) else "cpu")
if (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu))).to(device)
netG.load_state_dict(torch.load(f"{pth}/Gen.pt"))
netG.eval()

mask = load_mask('data/mask.tif', device)
unmask = load_mask('data/unmasked.tif', device)

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def optimise():
    noise = torch.randn(1, nz, lf, lf, lf, requires_grad=True, device=device)
    optimiser = torch.optim.Adam([noise], 0.001, (0.9,0.99))
    imgs = []
    iter = 0
    thresh = 0.05
    loss = thresh+1
    while loss>thresh:
        img = netG(noise)
        loss = pixel_wise_loss(img, mask, device=device)
        pw = loss.clone()
        loss = loss.mean()
        loss.backward()
        
        optimiser.step()
        noise.grad.zero_()
        if iter % 50 ==0:
            with torch.no_grad():
                print(loss.item())
                fig, ax = plt.subplots(2,3)
                fig.suptitle(iter)
                ax[0,0].set_title('Original')
                ax[0,1].set_title('Mask')
                ax[0,2].set_title('Original-masked')
                ax[1,0].set_title('G output')
                ax[1,1].set_title('Inpainted')
                ax[1,2].set_title('MSE loss')

                ax[0,0].imshow(post_process(unmask.unsqueeze(0))[0].permute(1,2,3,0).cpu()[32])
                ax[0,1].imshow(mask[-1, 32].cpu())
                ax[0,2].imshow(post_process(mask.unsqueeze(0))[0,0:3].permute(1,2,3,0).cpu()[32])
                ax[1,0].imshow(post_process(img)[0].permute(1,2,3,0).cpu()[32])
                ax[1,1].imshow(post_process(inpaint(img, unmask))[0].permute(1,2,3,0).cpu()[32])
                ax[1,2].imshow(pw[0].permute(1,2,3,0).cpu()[32])
                fig.tight_layout()
                plt.savefig(f'optim/iter_{iter}.png')
                plt.close()
        iter +=1

            
    
    images = []
    filenames = os.listdir('optim')
    filenames = sorted_alphanumeric(filenames)
    try:
        filenames.remove('movie.gif')
    except:
        pass
    for filename in filenames:
        images.append(imageio.imread(f'optim/{filename}'))
    imageio.mimsave('optim/movie.gif', images, duration=0.5)
        
