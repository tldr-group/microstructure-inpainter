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
        self.save_inpaint = self.c.opt_iters//self.frames
        self.opt_whilst_train = not c.cli
        # self.opt_whilst_train = False


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
        
        if c.wandb:
            wandb_init(tag, offline=False)
            wandb.watch(netG)
            wandb.watch(netD)

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
            if c.wandb:
                wandb.log({'D_real': out_real.item(), 'D_fake': out_fake.item()}, step=i)

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
            if i%100 == 0:
                
                torch.save(netG.state_dict(), f'{path}/Gen.pt')
                torch.save(netD.state_dict(), f'{path}/Disc.pt')
                if ('cuda' in str(device)) and (ngpu > 1):
                    end_overall.record()
                    torch.cuda.synchronize()
                    t = start_overall.elapsed_time(end_overall)
                else:
                    end_overall = time.time()
                    t = end_overall-start_overall
                if self.opt_whilst_train:
                    mse, img, inpaint = self.inpaint(netG, device=device, opt_iters=self.opt_iters)

                    if c.cli:
                        print(f'Iter: {i}, Time: {t:.1f}, MSE: {mse:.2g}, Wass: {abs(wass.item()):.2g}')
                        if c.wandb:
                            wandb.log({'mse':mse, 'wass':wass.item(), 
                            'gp': gradient_penalty.item(), 
                            'raw out': wandb.Image(img.cpu().numpy()), 'inpaint out': wandb.Image(inpaint)}, step=i)
                    else:
                        self.progress.emit(i, t, mse, abs(wass.item()))
                    
                    time_list.append(t)
                    mses.append(mse)
                    wass_list.append(abs(wass.item()))
                    iter_list.append(i)
                    df = pd.DataFrame({'MSE': mses, 'iters': iter_list, 'mse': mses, 'time': time_list, 'wass': wass_list})
                    df.to_pickle(f'runs/{tag}/metrics.pkl')
                else:
                    print(f'Iter: {i}, Time: {t:.1f}')

                

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

    def inpaint(self, netG, save_path=None, border=False, device='cpu', opt_iters=1000):
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
            # x1 += 32 - w%32
            # y1 += 32 - h%32
            w, h = x1-x0, y1-y0
            im_crop = img[:, x0-16:x1+16, y0-16:y1+16]
            mask_crop = self.mask[x0-16:x1+16, y0-16:y1+16]
            c, w, h = im_crop.shape
            if self.c.conv_resize:
                lx, ly = int(w/16), int(h/16)
            else:
                lx, ly = int(w/32) + 2, int(h/32) + 2
            inpaints, mse, raw = self.optimise_noise(lx, ly, im_crop, mask_crop, netG, device, opt_iters=opt_iters)
            for fimg, inpaint in enumerate(inpaints):
                final_imgs[fimg][x0:x1,  y0:y1] = inpaint
        for i, final_img in enumerate(final_imgs):
            if self.c.image_type=='n-phase':
                final_img[self.mask==0] = final_img_fresh[self.mask==0]
                final_img = (final_img.numpy()/final_img.max())
                out = np.stack([final_img for i in range(3)], -1)
                plt.imsave(f'data/temp/temp{i}.png', out)
            else:
                for ch in range(self.c.n_phases):    
                    final_img[:,:,ch][self.mask==0] = final_img_fresh[:,:,ch][self.mask==0]
                if self.c.image_type=='colour':
                    out = final_img.numpy()
                    plt.imsave(f'data/temp/temp{i}.png', out)
                elif self.c.image_type=='grayscale':
                    out = np.concatenate([final_img for i in range(3)], -1)
                    plt.imsave(f'data/temp/temp{i}.png', out)
        if save_path:
            fig, ax = plt.subplots()
            final_img = final_imgs[-1]
            final_img[self.mask==0] = final_img_fresh[self.mask==0]
            final_img = (final_img.numpy()/final_img.max())
            if self.c.image_type == 'n-phase':
                # plot in rainbow colours for n_phase
                x,y = final_img.shape
                phases = torch.unique(final_img)
                color = iter(cm.get_cmap(self.c.cm)(np.linspace(0, 1, self.c.n_phases)))
                out = torch.zeros((3, x, y))
                for i, ph in enumerate(phases):
                    col = next(color)
                    col = torch.tile(torch.Tensor(col[0:3]).unsqueeze(1).unsqueeze(1), (x,y))
                    out = torch.where((final_img == ph), col, out)
                out = out.permute(1,2,0)
                ax.imshow(out.numpy())
                rect_col = '#CC2825'
                
            elif self.c.image_type=='grayscale':
                out = final_img
                rect_col = '#CC2825'
                ax.imshow(out.numpy(), cmap='gray')
            else:
                out = final_img
                ax.imshow(out.numpy())
                rect_col = '#CC2825'
            
            ax.set_axis_off()
            if border:
                rect = Rectangle((y0,x0),h_init, w_init,linewidth=1, ls='--', edgecolor=rect_col,facecolor='none')
                ax.add_patch(rect)
            plt.tight_layout()
            plt.savefig(f'{save_path}_border.png', transparent=True)
            ax.patches=[]
            plt.savefig(f'{save_path}.png', transparent=True)
        return mse, raw, out

    def optimise_noise(self, lx, ly, img, mask, netG, device, opt_iters=1000):
        if self.verbose:
            print(f'Optimisation params - iters: {opt_iters}, lr: {self.c.opt_lr}, kl_coeff: {self.c.opt_kl_coeff}')     
        target = img.to(device)
        for ch in range(self.c.n_phases):
            target[ch][mask==1] = -1
        # plt.imsave('test2.png', torch.cat([target.permute(1,2,0) for i in range(3)], -1).cpu().numpy())
        # plt.imsave('test.png', np.stack([mask for i in range(3)], -1))
        bs = 1
        target = target.unsqueeze(0)
        target = torch.tile(target, (bs, 1, 1,1))
        multi = torch.ones_like(target)
        for i in range(multi.shape[-2]):
            for j in range(multi.shape[-1]):
                multi[:,:,i,j] = max(abs(i-multi.shape[-2]//2+1),abs(j-multi.shape[-1]//2+1))
        multi = abs(multi-multi.max())
        noise = [torch.nn.Parameter(torch.randn(bs, self.c.nz, lx, ly, requires_grad=True, device=device))]
        opt = torch.optim.Adam(params=noise, lr=self.c.opt_lr)
        kl_loss = nn.KLDivLoss(reduction="batchmean")
        inpaints = []
        mses = []
        if opt_iters>0:
            for i in range(opt_iters):
                raw = netG(noise[0])
                loss = (raw - target)**2
                loss[target==-1] = 0
                loss_copy = loss.clone()
                loss = loss.mean()
                loss.backward()
                opt.step()
                mses.append(loss_copy.mean(dim=(1,2,3)).min().detach().cpu())
                # kl divergence as part of loss to enforce random normal distribution
                loss_kl = self.c.opt_kl_coeff*kl_loss(noise[0], torch.randn_like(noise[0]))
                loss_kl.backward()
                opt.step()
                with torch.no_grad():
                    # normalise noise
                    for b in range(bs):
                        noise[0][b] -= torch.tile(torch.mean(noise[0][b], dim=[0]), (self.c.nz,1,1))
                        noise[0][b] /= torch.tile(torch.std(noise[0][b], dim=[0]), (self.c.nz,1,1))
                if i%self.save_inpaint==0:
                    if self.c.image_type == 'n-phase':
                        raw = torch.argmax(raw[0], dim=0)[16:-16, 16:-16].detach().cpu()
                    else:
                        raw = raw[0].permute(1,2,0)[16:-16, 16:-16].detach().cpu()
                    inpaints.append(raw)
                        
        else:
            loss = torch.Tensor([0])
            raw = netG(noise[0])
        return inpaints, loss.item(), raw
    
    def generate(self, save_path=None, border=False, opt_iters=None):
        if opt_iters==None:
            opt_iters=self.c.opt_iters
        self.save_inpaint = opt_iters//self.frames
        if self.verbose:
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

        mse, raw, _ = self.inpaint(netG, save_path=save_path, border=border, device=device, opt_iters=opt_iters)

        return raw
