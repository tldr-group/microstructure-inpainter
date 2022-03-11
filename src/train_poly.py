from src.util import *
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
import tifffile
import time
from PyQt5.QtCore import QObject, pyqtSignal


class PolyWorker(QObject):
    def __init__(self, c, netG, netD, real_seeds, mask, poly_rects, overwrite):
        super().__init__()
        self.c = c
        self.netG = netG
        self.netD = netD
        self.real_seeds = real_seeds
        self.poly_rects = poly_rects
        self.mask = mask
        self.overwrite = overwrite
    finished = pyqtSignal()
    progress = pyqtSignal(int)

    def train(self):
        """[summary]

        :param c: [description]
        :type c: [type]
        :param Gen: [description]
        :type Gen: [type]
        :param Disc: [description]
        :type Disc: [type]
        :param offline: [description], defaults to True
        :type offline: bool, optional
        """

        # Assign torch device
        c = self.c 
        netG = self.netG 
        netD = self.netD
        real_seeds = self.real_seeds
        poly_rects = self.poly_rects 
        mask = self.mask
        overwrite = self.overwrite 
        offline=True
        ngpu = c.ngpu 
        tag = c.tag
        path = c.path
        device = torch.device(c.device_name if(
            torch.cuda.is_available() and ngpu > 0) else "cpu")
        print(f'Using {ngpu} GPUs')
        print(device, " will be used.\n")
        cudnn.benchmark = True

        # Get train params
        l, batch_size, beta1, beta2, num_epochs, iters, lrg, lr, Lambda, critic_iters, lz, nz, = c.get_train_params()

        # Read in data
        training_imgs, nc = preprocess(c.data_path)

        # Define Generator network
        netG = netG().to(device)
        netD = netD().to(device)
        if ('cuda' in str(device)) and (ngpu > 1):
            netD = (nn.DataParallel(netD, list(range(ngpu)))).to(device)
            netG = nn.DataParallel(netG, list(range(ngpu))).to(device)
        optD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))
        optG = optim.Adam(netG.parameters(), lr=lrg, betas=(beta1, beta2))

        if not overwrite:
            netG.load_state_dict(torch.load(f"{path}/Gen.pt"))
            netD.load_state_dict(torch.load(f"{path}/Disc.pt"))

        wandb_init(tag, offline)
        wandb.watch(netD, log='all', log_freq=100)
        wandb.watch(netG, log='all', log_freq=100)

        for epoch in range(num_epochs):
            times = []
            for i in range(iters):
                # Discriminator Training
                if ('cuda' in str(device)) and (ngpu > 1):
                    start_overall = torch.cuda.Event(enable_timing=True)
                    end_overall = torch.cuda.Event(enable_timing=True)
                    start_overall.record()
                else:
                    start_overall = time.time()

                netD.zero_grad()

                noise = torch.randn(batch_size, nz, lz, lz, device=device)
                fake_data = netG(noise).detach()

                real_data = batch_real_poly(training_imgs, l, batch_size, real_seeds).to(device)

                # Train on real
                out_real = netD(real_data).mean()
                # train on fake images
                out_fake = netD(fake_data).mean()
                gradient_penalty = calc_gradient_penalty(netD, real_data, fake_data, batch_size, l, device, Lambda, nc)

                # Compute the discriminator loss and backprop
                disc_cost = out_fake - out_real + gradient_penalty
                disc_cost.backward()

                optD.step()

                # Log results
                if not offline:
                    wandb.log({'Gradient penalty': gradient_penalty.item()})
                    wandb.log({'Wass': out_real.item() - out_fake.item()})
                    wandb.log({'Discriminator real': out_real.item()})
                    wandb.log({'Discriminator fake': out_fake.item()})

                # Generator training
                if i % int(critic_iters) == 0:
                    netG.zero_grad()
                    noise = torch.randn(batch_size, nz, lz, lz, device=device)
                    # Forward pass through G with noise vector
                    fake_data = netG(noise)
                    output = -netD(fake_data).mean()

                    # Calculate loss for G and backprop
                    output.backward()
                    optG.step()

                if ('cuda' in str(device)) and (ngpu > 1):
                    end_overall.record()
                    torch.cuda.synchronize()
                    times.append(start_overall.elapsed_time(end_overall))
                else:
                    end_overall = time.time()
                    times.append(end_overall-start_overall)


                # Every 50 iters log images and useful metrics
                if i % 100 == 0:
                    
                    torch.save(netG.state_dict(), f'{path}/Gen.pt')
                    torch.save(netD.state_dict(), f'{path}/Disc.pt')
                    # wandb_save_models(f'{path}/Disc.pt')
                    # wandb_save_models(f'{path}/Gen.pt')
                    self.inpaint(netG)
                    self.progress.emit(i)
                    times = []

        self.finish.emit()

    def inpaint(self, netG):
        img = preprocess(self.c.data_path)[0]
        final_img = torch.argmax(img, dim=0)
        final_img_fresh = torch.argmax(img, dim=0)
        print(f'inpainting {len(self.poly_rects)} patches')
        for rect in self.poly_rects:
            x0, y0, x1, y1 = (int(i) for i in rect)
            w, h = x1-x0, y1-y0
            x1 += 32 - w%32
            y1 += 32 - h%32
            w, h = x1-x0, y1-y0
            im_crop = img[:, x0-16:x1+16, y0-16:y1+16]
            mask_crop = self.mask[x0-16:x1+16, y0-16:y1+16]
            c, w, h = im_crop.shape
            lx, ly = int(w/32) + 2, int(h/32) + 2
            inpaint = self.optimise_noise(lx, ly, im_crop, mask_crop, netG)
            final_img[x0:x1,  y0:y1] = inpaint
        final_img[self.mask==0] = final_img_fresh[self.mask==0]
        final_img = (final_img.numpy()/final_img.max())
        plt.imsave(f'data/temp.png', np.stack([final_img for i in range(3)], -1))
        
    def optimise_noise(self, lx, ly, img, mask, netG):
        netG.eval()
        target = img.cuda()
        device = torch.device("cuda:0" if(
            torch.cuda.is_available() and self.c.ngpu > 0) else "cpu")
        target[:, mask] = -1
        target = target.unsqueeze(0)
        noise = [torch.nn.Parameter(torch.randn(1, self.c.nz, lx, ly, requires_grad=True, device=device))]
        opt = torch.optim.SGD(params=noise, lr=1)
        iters=200
        for i in range(iters):
            raw = netG(noise[0])
            loss = (raw - target)**2
            loss[target==-1] = 0
            loss = loss.mean()
            loss.backward()
            opt.step()
            
        netG.train()
        raw = torch.argmax(raw[0], dim=0)
        return raw[16:-16, 16:-16].detach().cpu()