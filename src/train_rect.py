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
        
    finished = pyqtSignal()
    progress = pyqtSignal(int, int, float)

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

        # Get train params
        l, dl, batch_size, beta1, beta2, num_epochs, iters, lrg, lr, Lambda, critic_iters, lz, nz, = c.get_train_params()

        mask = mask.to(device)
        unmasked = unmasked.to(device)
        # init noise
        noise = torch.nn.Parameter(torch.randn(batch_size, nz, c.seed_x, c.seed_y, requires_grad=True, device=device))
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
        i=0
        epoch = 0
        if c.cli:
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
            
        while not self.quit_flag and not converged:
            # Discriminator Training
            netD.zero_grad()

            d_noise = make_noise(noise.detach(), c.seed_x, c.seed_y, c, device)
            fake_data = netG(d_noise).detach()
            fake_data = crop(fake_data,dl)
            real_data = batch_real(training_imgs, dl, batch_size, c.mask_coords).to(device)
            # Train on real
            out_real_batch = netD(real_data)
            out_real = out_real_batch.mean()
            # train on fake images
            out_fake = netD(fake_data).mean()
            gradient_penalty = calc_gradient_penalty(netD, real_data, fake_data, batch_size, dl, device, Lambda, nc)

            # Compute the discriminator loss and backprop
            wass = out_fake - out_real
            disc_cost = wass + gradient_penalty
            disc_cost.backward()

            optD.step()

            # Generator training
            if (i % int(critic_iters)) == 0:
                netG.zero_grad()
                optNoise.zero_grad()
                noise_G = make_noise(noise, c.seed_x, c.seed_y, c, device)
                # Forward pass through G with noise vector
                fake_data = netG(noise_G)
                output = -netD(crop(fake_data, dl)).mean()
                pw = pixel_wise_loss(fake_data, mask, device=device)
                output += pw*c.pw_coeff
                # Calculate loss for G and backprop
                output.backward(retain_graph=True)
                optG.step()
                optNoise.step()
                with torch.no_grad():
                    noise -= torch.tile(torch.mean(noise, dim=[1]).unsqueeze(1), (1, nz,1,1))
                    noise /= torch.tile(torch.std(noise, dim=[1]).unsqueeze(1), (1, nz,1,1))


            # Every 50 iters log images and useful metrics
            if i % 50 == 0:
                with torch.no_grad():
                    torch.save(netG.state_dict(), f'{path}/Gen.pt')
                    torch.save(netD.state_dict(), f'{path}/Disc.pt')
                    torch.save(noise, f'{path}/noise.pt')

                    plot_noise = make_noise(noise.detach().clone(), c.seed_x, c.seed_y, c, device)[0].unsqueeze(0)
                    img = netG(plot_noise).detach()

                    pixmap = update_pixmap_rect(training_imgs, img, c)
                    # tpc_loss = two_pc_metric(fake_data.detach().cpu(), tpc_real)
                    # boundary_D = evaluate_D_on_boundary(netD, fake_data.detach(), c.dl, device)
                    self.progress.emit(i, epoch, pw.item())
                    if ('cuda' in str(device)) and (ngpu > 1):
                        end_overall.record()
                        torch.cuda.synchronize()
                        t = start_overall.elapsed_time(end_overall)
                    else:
                        end_overall = time.time()
                        t = end_overall-start_overall
                    if c.cli:
                        print(f'Iter: {i}, Time: {t:.1f}, MSE: {pw.item():.2g}, Wass: {abs(wass.item()):.2g}')
                        time_list.append(t)
                        mses.append(pw.item())
                        wass_list.append(abs(wass.item()))
                        iter_list.append(i)
                        df = pd.DataFrame({'MSE': mses, 'iters': iter_list, 'mse': mses, 'time': time_list, 'wass': wass_list})
                        df.to_pickle(f'runs/{tag}/metrics.pkl')
                    
                    converged_list.append(check_convergence(pw.item(), abs(wass.item())))
                    if np.sum(converged_list[-4:])==4:
                        print("Training Converged")
                        converged=True
            i+=1
            if i%iters==0:
                epoch +=1
            if self.quit_flag:
                self.finished.emit()
                print("TRAINING QUTTING")
        self.finished.emit()
        print("TRAINING FINISHED")        
    
    def generate(self):
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
        with torch.no_grad():
            idx = np.random.randint(self.c.batch_size)
            plot_noise = make_noise(noise.detach().clone(), self.c.seed_x, self.c.seed_y, self.c, device)[idx].unsqueeze(0)
            img = netG(plot_noise).detach()
            update_pixmap_rect(self.training_imgs, img, self.c)
        