from src.util import *
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
import tifffile
import time
from PyQt5.QtCore import QObject, pyqtSignal
from copy import deepcopy

class PolyWorker(QObject):
    def __init__(self, c, netG, netD, real_seeds, mask, poly_rects, frames, overwrite):
        super().__init__()
        self.c = c
        self.netG = netG
        self.netD = netD
        self.real_seeds = real_seeds
        self.poly_rects = poly_rects
        self.mask = mask
        self.overwrite = overwrite
        self.frames = frames
        self.quit_flag = False
        self.opt_iters = 1000
        self.save_inpaint = self.opt_iters//self.frames


    finished = pyqtSignal()
    progress = pyqtSignal(int, int, float, float)

    def stop(self):
        self.quit_flag = True

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
        l, batch_size, beta1, beta2, lrg, lr, Lambda, critic_iters, lz, nz, = c.get_train_params()

        # Read in data
        training_imgs, nc = preprocess(c.data_path, c.image_type)

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


        i=0
        t = 0
        mses = []
        iter_list = []
        time_list = []
        wass_list = []

        # start timing training
        if ('cuda' in str(device)) and (ngpu > 1):
                start_overall = torch.cuda.Event(enable_timing=True)
                end_overall = torch.cuda.Event(enable_timing=True)
                start_overall.record()
        else:
            start_overall = time.time()
        
        converged = False
        converged_list = []

        while not self.quit_flag and not converged and t<c.timeout and i<c.max_iters:
            # Discriminator Training
            netD.zero_grad()

            noise = torch.randn(batch_size, nz, lz, lz, device=device)
            fake_data = netG(noise).detach()
            real_data = batch_real_poly(training_imgs, l, batch_size, real_seeds).to(device)
            # Train on real
            out_real = netD(real_data).mean()
            # train on fake images
            out_fake = netD(fake_data).mean()
            gradient_penalty = calc_gradient_penalty(netD, real_data, fake_data, batch_size, l, device, Lambda, nc)
            wass = out_fake - out_real
            # Compute the discriminator loss and backprop
            disc_cost = wass + gradient_penalty
            disc_cost.backward()

            optD.step()

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


            # Every 50 iters log images and useful metrics
            if i%50 == 0:
                
                torch.save(netG.state_dict(), f'{path}/Gen.pt')
                torch.save(netD.state_dict(), f'{path}/Disc.pt')
                mse, _ = self.inpaint(netG, device=device)

                if ('cuda' in str(device)) and (ngpu > 1):
                    end_overall.record()
                    torch.cuda.synchronize()
                    t = start_overall.elapsed_time(end_overall)
                else:
                    end_overall = time.time()
                    t = end_overall-start_overall

                # Normalise wass for comparison (this needs thinking about more!)
                # TODO
                wass = wass / np.prod(real_data.shape)

                if c.cli:
                    print(f'Iter: {i}, Time: {t:.1f}, MSE: {mse:.2g}, Wass: {abs(wass.item()):.2g}')
                else:
                    self.progress.emit(i, t, mse, abs(wass.item()))
                
                time_list.append(t)
                mses.append(mse)
                wass_list.append(abs(wass.item()))
                iter_list.append(i)
                df = pd.DataFrame({'MSE': mses, 'iters': iter_list, 'mse': mses, 'time': time_list, 'wass': wass_list})
                df.to_pickle(f'runs/{tag}/metrics.pkl')
                

            i+=1
            if i==c.max_iters:
                print(f"Max iterations reached: {i}")
            if self.quit_flag:
                self.finished.emit()
                print("TRAINING QUITTING")
        if t>c.timeout:
            print(f"Timeout: {t:.2g}") 
        self.finished.emit()
        print("TRAINING FINISHED")

    def inpaint(self, netG, save_path=None, border=False, device='cpu'):
        img = preprocess(self.c.data_path, self.c.image_type)[0]
        
        if self.c.image_type =='n-phase':
            final_imgs = [torch.argmax(img, dim=0) for i in range(self.frames)]
            final_img_fresh = torch.argmax(img, dim=0)
        else:
            final_img_fresh = img.permute(1, 2, 0)
            final_imgs = [deepcopy(img.permute(1, 2, 0)) for i in range(self.frames)]
        for rect in self.poly_rects:
            x0, y0, x1, y1 = (int(i) for i in rect)
            w, h = x1-x0, y1-y0
            w_init, h_init = w,h
            x1 += 32 - w%32
            y1 += 32 - h%32
            w, h = x1-x0, y1-y0
            im_crop = img[:, x0-16:x1+16, y0-16:y1+16]
            mask_crop = self.mask[x0-16:x1+16, y0-16:y1+16]
            c, w, h = im_crop.shape
            if self.c.conv_resize:
                lx, ly = int(w/16), int(h/16)
            else:
                lx, ly = int(w/32) + 2, int(h/32) + 2
            inpaints, mse, raw = self.optimise_noise(lx, ly, im_crop, mask_crop, netG, device)
            for fimg, inpaint in enumerate(inpaints):
                final_imgs[fimg][x0:x1,  y0:y1] = inpaint
        for i, final_img in enumerate(final_imgs):
            if self.c.image_type=='n-phase':
                final_img[self.mask==0] = final_img_fresh[self.mask==0]
                final_img = (final_img.numpy()/final_img.max())
                plt.imsave(f'data/temp/temp{i}.png', np.stack([final_img for i in range(3)], -1))
            else:
                for ch in range(self.c.n_phases): 
                    final_img[:,:,ch][self.mask==0] = final_img_fresh[:,:,ch][self.mask==0]
                if self.c.image_type=='colour':
                    plt.imsave(f'data/temp/temp{i}.png', final_img.numpy())
                elif self.c.image_type=='grayscale':
                    plt.imsave(f'data/temp/temp{i}.png', np.concatenate([final_img for i in range(3)], -1))
        if save_path:
            fig, ax = plt.subplots()
            final_img = final_imgs[-1]
            final_img[self.mask==0] = final_img_fresh[self.mask==0]
            final_img = (final_img.numpy()/final_img.max())
            ax.imshow(np.stack([final_img for i in range(3)], -1))
            ax.set_axis_off()
            if border:
                rect = Rectangle((x0,y0),w_init,h_init,linewidth=2,edgecolor='b',facecolor='none')
                ax.add_patch(rect)
            plt.savefig(save_path, transparent=True)
        return mse, raw

    def optimise_noise(self, lx, ly, img, mask, netG, device):
              
        target = img.to(device)
        for ch in range(self.c.n_phases):
            target[ch][mask==1] = -1
        # plt.imsave('test2.png', torch.cat([target.permute(1,2,0) for i in range(3)], -1).cpu().numpy())
        # plt.imsave('test.png', np.stack([mask for i in range(3)], -1))

        target = target.unsqueeze(0)
        noise = [torch.nn.Parameter(torch.randn(1, self.c.nz, lx, ly, requires_grad=True, device=device))]
        opt = torch.optim.Adam(params=noise, lr=0.005)
        inpaints = []
        for i in range(self.opt_iters):
            raw = netG(noise[0])
            loss = (raw - target)**2
            loss[target==-1] = 0
            loss = loss.sum() / ((target!=-1).sum()*loss.shape[1]*loss.shape[0])
            # Isaac to Steve - is this MSE an average over only the pixels that are valid? i.e. sum of loss / #pixels in the mask?
            loss.backward()
            opt.step()
            with torch.no_grad():
                noise[0] -= torch.tile(torch.mean(noise[0], dim=[1]), (1, self.c.nz,1,1))
                noise[0] /= torch.tile(torch.std(noise[0], dim=[1]), (1, self.c.nz,1,1))
            if i%self.save_inpaint==0:
                if self.c.image_type == 'n-phase':
                    raw = torch.argmax(raw[0], dim=0)[16:-16, 16:-16].detach().cpu()
                else:
                    raw = raw[0].permute(1,2,0)[16:-16, 16:-16].detach().cpu()
                inpaints.append(raw)
        return inpaints, loss.item(), raw
    
    def generate(self, save_path=None, border=False):
        print("Generating new inpainted image")
        device = torch.device(self.c.device_name if(
            torch.cuda.is_available() and self.c.ngpu > 0) else "cpu")
        netG = self.netG().to(device)
        netD = self.netD().to(device)
        if ('cuda' in str(device)) and (self.c.ngpu > 1):
            netD = (nn.DataParallel(netD, list(range(self.c.ngpu)))).to(device)
            netG = nn.DataParallel(netG, list(range(self.c.ngpu))).to(device)
        netG.load_state_dict(torch.load(f"{self.c.path}/Gen.pt"))
        netD.load_state_dict(torch.load(f"{self.c.path}/Disc.pt"))

        mse, raw = self.inpaint(netG, save_path=save_path, border=border, device=device)

        return raw
