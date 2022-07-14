import pandas as pd
from src.util import *
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np
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

        print(f"Data shape: {training_imgs.shape}")

        # Get train params
        l, dl, batch_size, beta1, beta2, lrg, lr, Lambda, critic_iters, lz, nz, = c.get_train_params()

        pw_iters = 0

        mask = mask.to(device)
        unmasked = unmasked.to(device)
        # init noise
        noise = torch.nn.Parameter(init_noise(batch_size, nz, c, device))
        optNoise = torch.optim.Adam([noise], lr=0.01,betas=(beta1, beta2))
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
            wandb_init(tag, offline=False)
            wandb.watch(netG)
            wandb.watch(netD)
        i=0
        t=0
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
            
        while not self.quit_flag and t<c.timeout and i<c.max_iters:
            # Discriminator Training
            netD.zero_grad()
            netG.train()

            mask_noise = True

            d_noise = make_noise(noise.detach(), c.seed_x, c.seed_y, c, device, mask_noise=mask_noise)
            fake_data = netG(d_noise).detach()
            fake_data = crop(fake_data,dl)
            real_data = batch_real(training_imgs, dl, batch_size, c.mask_coords).to(device)
            # Train on real
            out_real = netD(real_data).mean()
            # train on fake images
            out_fake = netD(fake_data).mean()
            gradient_penalty = calc_gradient_penalty(netD, real_data, fake_data, batch_size, dl, device, Lambda, nc)

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
                optNoise.zero_grad()
                noise_G = make_noise(noise, c.seed_x, c.seed_y, c, device, mask_noise=mask_noise)
                
                # Forward pass through G with noise vector
                fake_data = netG(noise_G)
                # output = -netD(crop(fake_data, dl, shift=True, prob=0.05)).mean()
                output = -netD(crop(fake_data, dl)).mean()
                
                if i == pw_iters:
                    print(f"Turning on pixel wise loss, with coeff {c.pw_coeff}")
                if i >= pw_iters:
                    pw = pixel_wise_loss(fake_data, mask, unmasked, mode='mse', device=device)
                    # output += pw.sum()*np.min([i*1e-2, c.pw_coeff])
                    output += pw*c.pw_coeff
                else:
                    pw = torch.zeros_like((fake_data))

                 # # Calculate loss for G and backprop
                output.backward(retain_graph=True)
                optG.step()
                optNoise.step()

                with torch.no_grad():
                    noise -= torch.tile(torch.mean(noise, dim=[1]).unsqueeze(1), (1, nz,1,1))
                    noise /= torch.tile(torch.std(noise, dim=[1]).unsqueeze(1), (1, nz,1,1))

            # Every 50 iters log images and useful metrics
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
                        plot_noise = make_noise(noise.detach().clone(), c.seed_x, c.seed_y, c, device, mask_noise=mask_noise)[0].unsqueeze(0)
                        img = netG(plot_noise).detach()
                        # plt.imsave('test.png', img.permute(0,2,3,1).detach().cpu()[0,...,0].numpy())
                        pixmap = update_pixmap_rect(training_imgs, img, c)
                    
                        # Normalise wass for comparison
                        # wass = wass / np.prod(real_data.shape)

                        if c.cli:
                            print(f'Iter: {i}, Time: {t:.1f}, MSE: {pw.sum().item():.2g}, Wass: {abs(wass.item()):.2g}')
                            # if i>= miniD_iters:
                            #     print(f'MiniD Wass: {wass_miniD.item():.2g}')
                            if c.wandb:
                                wandb.log({'mse':pw.nanmean().item(), 'wass':wass.item(), 
                                'gp': gradient_penalty.item(), 
                                'raw out': wandb.Image(img[0,0].cpu()), 'inpaint out': wandb.Image(pixmap)}, step=i)
                                # 'mse image':wandb.Image(pw[0,0]/pw[0,0].max()),
                        else:
                            self.progress.emit(i, t, pw.item(), abs(wass.item()))
                        # save metrics to pkl
                        # time_list.append(t)
                        # mses.append(pw.sum().item())
                        # wass_list.append(abs(wass.item()))
                        # iter_list.append(i)
                        # df = pd.DataFrame({'MSE': mses, 'iters': iter_list, 'mse': mses, 'time': time_list, 'wass': wass_list})
                        # df.to_pickle(f'runs/{tag}/metrics.pkl')
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
    
    def train_alt(self):
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
        ngpu = c.ngpu
        tag = c.tag
        path = c.path
        device = torch.device(c.device_name if(
            torch.cuda.is_available() and ngpu > 0) else "cpu")
        print(f'Using {ngpu} GPUs')
        print(device, " will be used.\n")
        cudnn.benchmark = True

        
        # init noise
        # Define Generator network
        netG = Gen().to(device)
        netD = Disc().to(device)

        if ('cuda' in str(device)) and (ngpu > 1):
            netD = (nn.DataParallel(netD, list(range(ngpu)))).to(device)
            netG = nn.DataParallel(netG, list(range(ngpu))).to(device)
        if not overwrite:
            netG.load_state_dict(torch.load(f"{path}/Gen.pt"))
            netD.load_state_dict(torch.load(f"{path}/Disc.pt"))
            noise = torch.load(f'{c.path}/noise.pt')
        if c.wandb:
            wandb_init(tag, offline=False)
            wandb.watch(netG)
            wandb.watch(netD)

        netG, netD, i = self.initial_train(netG, netD, device)
        noise, optNoise = self.train_seed_opt(netG, device)

        self.finetune(noise, optNoise, i, netG, netD, device)

    def initial_train(self, netG, netD, device):
        c = self.c
        i=0
        t=0
        # Get train params
        l, dl, batch_size, beta1, beta2, lrg, lr, Lambda, critic_iters, lz, nz, = c.get_train_params()
        nc = self.nc
        mask = self.mask
        unmasked = self.unmasked
        training_imgs = self.training_imgs
        optD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))
        optG = optim.Adam(netG.parameters(), lr=lrg, betas=(beta1, beta2))
        path = c.path
        ngpu = c.ngpu
        mask = mask.to(device)
        unmasked = unmasked.to(device)

        # start timing training
        if ('cuda' in str(device)) and (ngpu > 1):
                start_overall = torch.cuda.Event(enable_timing=True)
                end_overall = torch.cuda.Event(enable_timing=True)
                start_overall.record()
        else:
            start_overall = time.time()

        self.start_overall = start_overall

        while not self.quit_flag and t<c.timeout and i<c.max_iters:
            # Discriminator Training
            netD.zero_grad()
            netG.train()

            # mask_noise = True

            # d_noise = make_noise(noise.detach(), c.seed_x, c.seed_y, c, device, mask_noise=mask_noise)
            d_noise = torch.randn(batch_size, nz, c.img_seed_x, c.img_seed_y).to(device)
            fake_data = netG(d_noise).detach()
            # fake_data = crop(fake_data,dl)
            real_data = batch_real(training_imgs, fake_data.shape[2], batch_size, c.mask_coords).to(device)
            # Train on real
            out_real = netD(real_data).mean()
            # train on fake images
            out_fake = netD(fake_data).mean()
            gradient_penalty = calc_gradient_penalty(netD, real_data, fake_data, batch_size, fake_data.shape[2], device, Lambda, nc)

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
                noise_G = torch.randn(batch_size, nz, c.img_seed_x, c.img_seed_y).to(device)
                # Forward pass through G with noise vector
                fake_data = netG(noise_G)
                output = -netD(fake_data).mean()
                output.backward(retain_graph=True)
                optG.step()

            # Every 50 iters log images and useful metrics
            if i % 100 == 0:
                netG.eval()
                with torch.no_grad():
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
                        plot_noise = torch.randn(batch_size, nz, c.img_seed_x, c.img_seed_y).to(device)
                        img = netG(plot_noise).detach()
                        plt.imsave('test.png', img.permute(0,2,3,1).detach().cpu()[0,...,0].numpy())
                        # pixmap = update_pixmap_rect(training_imgs, img, c)

                        if c.cli:
                            print(f'Iter: {i}, Time: {t:.1f}, Wass: {abs(wass.item()):.2g}')
                            # if i>= miniD_iters:
                            if c.wandb:
                                wandb.log({'wass':wass.item(), 
                                'gp': gradient_penalty.item(), 
                                'raw out': wandb.Image(img[0,0].cpu()),}, step=i)
                                # 'mse image':wandb.Image(pw[0,0]/pw[0,0].max()),
                        else:
                            self.progress.emit(i, t, abs(wass.item()))
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
        print("INITIAL TRAINING FINISHED")

        return netG, netD, i

    def train_seed_opt(self, netG, device):
        print("Beginning noise optimisation")
        c = self.c

        noise = torch.nn.Parameter(init_noise(1, c.nz, c, device))
        optNoise = torch.optim.Adam([noise], lr=0.01,betas=(c.beta1, c.beta2))

        for j in range(int(c.opt_iters)):
            optNoise.zero_grad()
            fake_data = netG(noise)
            pw = pixel_wise_loss(fake_data, self.mask.to(device), self.unmasked, mode='ce', device=device)
            pw.backward()
            optNoise.step()
        
            with torch.no_grad():
                    noise -= torch.tile(torch.mean(noise, dim=[1]), (1, c.nz,1,1))
                    noise /= torch.tile(torch.std(noise, dim=[1]), (1, c.nz,1,1))
            
            if j%50 == 0:
                print(f"Opt iter {j}")
        plt.imsave('search.png', fake_data.permute(0,2,3,1).detach().cpu()[0,...,0].numpy())
        torch.save(noise, f'{c.path}/noise.pt')
        return noise, optNoise

    def finetune(self, noise, optNoise, i, netG, netD, device):
        print("Finetuning")
        c = self.c
        # Get train params
        l, dl, batch_size, beta1, beta2, lrg, lr, Lambda, critic_iters, lz, nz, = c.get_train_params()
        nc = self.nc
        mask = self.mask
        unmasked = self.unmasked
        training_imgs = self.training_imgs
        optD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))
        optG = optim.Adam(netG.parameters(), lr=lrg, betas=(beta1, beta2))
        path = c.path
        ngpu = c.ngpu
        mask = mask.to(device)
        unmasked = unmasked.to(device)

        while not self.quit_flag and i<(c.finetune_iters+c.max_iters):
            # Discriminator Training
            netD.zero_grad()
            netG.train()

            # mask_noise = True

            d_noise = make_noise(noise.detach(), c.seed_x, c.seed_y, c, device)
            fake_data = crop(netG(d_noise).detach(), dl)
            # fake_data = crop(fake_data,dl)
            real_data = batch_real(training_imgs, dl, batch_size, c.mask_coords).to(device)
            # Train on real
            out_real = netD(real_data).mean()
            # train on fake images
            out_fake = netD(fake_data).mean()
            gradient_penalty = calc_gradient_penalty(netD, real_data, fake_data, batch_size, dl, device, Lambda, nc)

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
                optNoise.zero_grad()
                noise_G = make_noise(noise, c.seed_x, c.seed_y, c, device)
                
                # Forward pass through G with noise vector
                fake_data = netG(noise_G)
                # output = -netD(crop(fake_data, dl, shift=True, prob=0.05)).mean()
                output = -netD(crop(fake_data, dl)).mean()
                
                pw = pixel_wise_loss(fake_data, mask, unmasked, mode='ce', device=device)
                output += pw*c.pw_coeff

                 # # Calculate loss for G and backprop
                output.backward(retain_graph=True)
                optG.step()
                optNoise.step()

                with torch.no_grad():
                    noise -= torch.tile(torch.mean(noise, dim=[1]).unsqueeze(1), (1, nz,1,1))
                    noise /= torch.tile(torch.std(noise, dim=[1]).unsqueeze(1), (1, nz,1,1))
            
            # Every 50 iters log images and useful metrics
            if i % 100 == 0:
                netG.eval()
                with torch.no_grad():
                    torch.save(netG.state_dict(), f'{path}/Gen.pt')
                    torch.save(netD.state_dict(), f'{path}/Disc.pt')
                    torch.save(noise, f'{c.path}/noise.pt')
                    
                    if ('cuda' in str(device)) and (ngpu > 1):
                        end_overall.record()
                        torch.cuda.synchronize()
                        t = self.start_overall.elapsed_time(end_overall)
                    else:
                        end_overall = time.time()
                        t = end_overall-self.start_overall
                    if self.opt_whilst_train:
                        plot_noise = make_noise(noise, c.seed_x, c.seed_y, c, device)
                        img = netG(plot_noise).detach()
                        plt.imsave('test.png', img.permute(0,2,3,1).detach().cpu()[0,...,0].numpy())
                        pixmap = update_pixmap_rect(training_imgs, img, c)

                        if c.cli:
                            print(f'Iter: {i}, Time: {t:.1f}, Wass: {abs(wass.item()):.2g}')
                            # if i>= miniD_iters:
                            if c.wandb:
                                wandb.log({'wass':wass.item(), 
                                'gp': gradient_penalty.item(), 
                                'raw out': wandb.Image(img[0,0].cpu()),
                                "pw": pw.item()*c.pw_coeff,
                                "inpaint": wandb.Image(pixmap)
                                }, step=i)
                                # 'mse image':wandb.Image(pw[0,0]/pw[0,0].max()),
                        else:
                            self.progress.emit(i, t, abs(wass.item()))
                    else:
                        print(f"Iter: {i}, Time {t:.1f}")
                    
            i+=1
            if i==c.finetune_iters+c.max_iters:
                print(f"Max iterations reached: {i}")
            if self.quit_flag:
                self.finished.emit()
                print("Quitting training")
        if t>c.timeout:
            print(f"Timeout: {t:.2g}")   
        self.finished.emit()
        print("TRAINING FINISHED")

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
        noise = torch.load(f'{self.c.path}/noise.pt')
        netG.eval()
        with torch.no_grad():
            idx = np.random.randint(self.c.batch_size)
            plot_noise = make_noise(noise.detach().clone(), self.c.seed_x, self.c.seed_y, self.c, device)[idx].unsqueeze(0)
            # plot_noise = torch.randn_like(plot_noise)
            img = netG(plot_noise).detach()
            plt.imsave('test.png', img.permute(0,2,3,1).detach().cpu()[0,...,0].numpy())
            f = update_pixmap_rect(self.training_imgs, img, self.c, save_path=save_path, border=border)
            if save_path:
                axs = f.axes
                f.savefig(f'{save_path}_border.png', transparent=True)
                for ax in axs:
                    ax.patches = []
                f.savefig(f'{save_path}.png', transparent=True)
        
            return img


        