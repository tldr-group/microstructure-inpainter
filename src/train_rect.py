from src.util import *
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch
import torch.nn as nn
import time
from PyQt5.QtCore import QObject, pyqtSignal

class RectWorker(QObject):
    def __init__(self, c, netG, netD, training_imgs, nc, mask=None, unmasked=None):
        super().__init__()
        self.c = c
        self.netG = netG
        self.netD = netD
        self.training_imgs = training_imgs
        self.nc = nc
        self.mask = mask
        self.unmasked = unmasked
        self.quit_flag = False
        # self.opt_whilst_train = not c.cli
        self.opt_whilst_train = True
        
    finished = pyqtSignal()
    progress = pyqtSignal(int, int, float, float)

    def stop(self):
        self.quit_flag = True


    def train(self, wandb=None):
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
        overwrite = True
        c = self.c
        Gen = self.netG
        Disc = self.netD
        training_imgs = self.training_imgs 
        nc = self.nc
        mask = self.mask
        unmasked = self.unmasked 
        ngpu = c.ngpu
        tag = c.tag
        path = c.path
        device = torch.device(c.device_name if(
            torch.cuda.is_available() and ngpu > 0) else "cpu")
        print(f'Using {ngpu} GPUs')
        print(device, " will be used.\n")
        cudnn.benchmark = True

        print(f"Data shape: {training_imgs.shape}. Inpainting shape: {c.mask_size} Seed size: {c.img_seed_x, c.img_seed_y}")

        # Get train params
        l, batch_size, beta1, beta2, lrg, lr, Lambda, critic_iters, nz, = c.get_train_params()

        mask = mask.to(device)
        unmasked = unmasked.to(device)
        # init noise
        noise = init_noise(1, nz, c, device)
        # Define Generator network
        netG = Gen().to(device)
        netD = Disc().to(device)

        if ('cuda' in str(device)) and (ngpu > 1):
            netD = (nn.DataParallel(netD, list(range(ngpu)))).to(device)
            netG = nn.DataParallel(netG, list(range(ngpu))).to(device)
        optD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))
        optG = optim.Adam(netG.parameters(), lr=lrg, betas=(beta1, beta2))
        if not overwrite:
            netG.load_state_dict(torch.load(f"{path}/Gen.pt"))
            netD.load_state_dict(torch.load(f"{path}/Disc.pt"))
            noise = torch.load(f'{c.path}/noise.pt')
        if c.wandb:
            wandb.wandb_init(tag, netG, netD, offline=False)
        i=0
        t=0
        # start timing training
        if ('cuda' in str(device)) and (ngpu > 1):
                start_overall = torch.cuda.Event(enable_timing=True)
                end_overall = torch.cuda.Event(enable_timing=True)
                start_overall.record()
        else:
            start_overall = time.time()
            
        while not self.quit_flag and t<c.timeout and i<c.max_iters:
            # Discriminator Training
            netD.zero_grad()
            netG.train()

            d_noise = torch.randn_like(noise).to(device)
            fake_data = netG(d_noise).detach()
            # fake_data = crop(fake_data,dl)
            real_data = batch_real(training_imgs, fake_data.shape[-2], fake_data.shape[-1], batch_size, c.mask_coords).to(device)
            # Train on real
            out_real = netD(real_data).mean()
            # train on fake images
            out_fake = netD(fake_data).mean()
            gradient_penalty = calc_gradient_penalty(netD, real_data, fake_data, batch_size, fake_data.shape[-2], fake_data.shape[-1], device, Lambda, nc)

            # Compute the discriminator loss and backprop
            wass = out_fake - out_real
            disc_cost = wass + gradient_penalty
            disc_cost.backward()

            optD.step()
            if c.wandb:
                wandb.log({'D_real': out_real.item(), 'D_fake': out_fake.item()}, step=i)

            # Generator training
            if (i % int(critic_iters)) == 0:
                netG.zero_grad()
                noise_G = torch.randn_like(noise).to(device)
                # Forward pass through G with noise vector
                fake_data = netG(noise_G)
                output = -netD(fake_data).mean()

                noise_G = make_noise(noise, device, mask_noise=True, delta=[-1,-1])
                fake_data = netG(noise_G)
                pw = pixel_wise_loss(fake_data, mask, unmasked, mode='mse', device=device)
                output += pw*c.pw_coeff
                 # # Calculate loss for G and backprop
                output.backward(retain_graph=True)
                optG.step()

            # Every 100 iters log images and useful metrics
            if i % 100 == 0:
                netG.eval()
                with torch.no_grad():
                    torch.save(netG.state_dict(), f'{path}/Gen.pt')
                    torch.save(netD.state_dict(), f'{path}/Disc.pt')
                    torch.save(noise, f'{path}/noise.pt')

                    
                    
                    if ('cuda' in str(device)) and (ngpu > 1):
                        end_overall.record()
                        torch.cuda.synchronize()
                        t = start_overall.elapsed_time(end_overall)
                    else:
                        end_overall = time.time()
                        t = end_overall-start_overall
                    if self.opt_whilst_train:
                        plot_noise = make_noise(noise.detach().clone(), device, mask_noise=True, delta=[-1,-1])
                        img = netG(plot_noise).detach()
                        pixmap = update_pixmap_rect(training_imgs, img, c)
                    

                        if c.cli:
                            print(f'Iter: {i}, Time: {t:.1f}, MSE: {pw.sum().item():.2g}, Wass: {abs(wass.item()):.2g}')
                            if c.wandb:
                                wandb.log({'mse':pw.nanmean().item(), 'wass':wass.item(), 
                                'gp': gradient_penalty.item(), 
                                'raw out': wandb.Image(img[0].cpu()), 'inpaint out': wandb.Image(pixmap)}, step=i)
                        else:
                            self.progress.emit(i, t, pw.item(), abs(wass.item()))
                    else:
                        print(f"Iter: {i}, Time {t:.1f}")
                    
            i+=1
            if i==c.max_iters:
                print(f"Max iterations reached: {i}")
            if self.quit_flag:
                self.finished.emit()
                print("Quitting training")
        if t>c.timeout:
            print(f"Timeout: {t:.2g}")   
        self.finished.emit()
        print("TRAINING FINISHED")        

    def generate(self, save_path=None, border=False, delta=None):
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
        noise = torch.load(f'{self.c.path}/noise.pt')
        netG.eval()
        with torch.no_grad():
            # delta is an int that dictates how much of the centre of the seed is random
            if delta is None:
                if min(noise.shape[2:])<10:
                    mask_noise=False
                else:
                    delta = torch.tensor(noise.shape[2:])-10
                    mask_noise=True
            elif delta=='rand':
                mask_noise=False
            plot_noise = make_noise(noise.detach().clone(), device, mask_noise=mask_noise, delta=delta)
            img = netG(plot_noise).detach()
            f = update_pixmap_rect(self.training_imgs, img, self.c, save_path=save_path, border=border)
            if save_path:
                axs = f.axes
                f.savefig(f'{save_path}_border.png', transparent=True)
                for ax in axs:
                    ax.patches = []
                f.savefig(f'{save_path}.png', transparent=True)
            return img


        