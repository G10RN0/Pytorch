import torch
import torch.nn as nn
import torch.optim as optim
from dataset import HorseZebraDataset
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import config
import sys
from tqdm import tqdm
from torchvision.utils import save_image
from model import Discriminator, Generator

def train(disc_h, disc_z, gen_h, gen_z, loader, opt_disc, opt_gen, F_loss_L1, F_loss_mse, gen_scaler, disc_scaler, epcho):

    for idx, (zebra, horse) in enumerate(tqdm(loader, leave=True)):
        zebra = zebra.to(config.device)
        horse = horse.to(config.device)

        #train Discryminator H and Z
        with torch.cuda.amp.autocast():
            #horse model training
            fake_horse = gen_h(zebra)
            D_H_real = disc_h(horse)
            D_H_fake = disc_h(fake_horse.detach())

            D_H_real_loss = F_loss_mse(D_H_real, torch.ones_like(D_H_real))
            D_H_fake_loss = F_loss_mse(D_H_fake, torch.zeros_like(D_H_fake))
            D_H_loss = D_H_real_loss + D_H_fake_loss

            #zebra model training
            fake_zebra = gen_z(horse)
            D_Z_real = disc_z(zebra)
            D_Z_fake = disc_z(fake_zebra.detach())

            D_Z_real_loss = F_loss_mse(D_Z_real, torch.ones_like(D_Z_real))
            D_Z_fake_loss = F_loss_mse(D_Z_fake, torch.zeros_like(D_Z_fake))
            D_Z_loss = D_Z_real_loss + D_Z_fake_loss

            #put it together
            D_loss = (D_Z_loss + D_H_loss)/2
        
        opt_disc.zero_grad()
        disc_scaler.scale(D_loss).backward()
        disc_scaler.step(opt_disc)
        disc_scaler.update()

        #train Generator H and Z
        with torch.cuda.amp.autocast():
            #adversiarial loss for both genarators
            D_H_fake = disc_h(fake_horse)
            D_Z_fake = disc_z(fake_zebra)

            loss_G_Z = F_loss_mse(D_H_fake, torch.ones_like(D_H_fake))
            loss_G_H = F_loss_mse(D_Z_fake, torch.ones_like(D_Z_fake))

            #cycle loss
            cycle_zebra = gen_z(fake_horse)
            cycle_horse = gen_h(fake_zebra)
            cycle_zebra_loss = F_loss_L1(zebra, cycle_zebra)
            cycle_horse_loss = F_loss_L1(horse, cycle_horse)

            #all togther
            G_loss = (loss_G_Z + loss_G_H + cycle_zebra_loss*config.lambda_cycle + cycle_horse_loss*config.lambda_cycle)

        opt_gen.zero_grad()
        gen_scaler.scale(G_loss).backward()
        gen_scaler.step(opt_gen)
        gen_scaler.update()

        if idx % 200 == 0:
            save_image(fake_horse*0.5+0.5, f'saved_images/horse_{idx}_{epcho}.png')
            save_image(fake_zebra*0.5+0.5, f'saved_images/zebra_{idx}_{epcho}.png')

def main():
    disc_h = Discriminator(3, 64).to(config.device)
    disc_z = Discriminator(3, 64).to(config.device)
    gen_h = Generator(3, 64).to(config.device)
    gen_z = Generator(3, 64).to(config.device)

    opt_disc = optim.Adam(
        list(disc_z.parameters()) + list(disc_h.parameters()),
        lr=config.learning_rate,
        betas=(0.5, 0.999)
    )

    opt_gen = optim.Adam(
        list(gen_z.parameters()) + list(gen_h.parameters()),
        lr=config.learning_rate,
        betas=(0.5, 0.999)
    )

    F_loss_L1 = nn.L1Loss()
    F_loss_mse = nn.MSELoss()

    dataset = HorseZebraDataset(root_h=config.train_dir+'/horse', root_z=config.train_dir+'/zebra', transform=config.transforms)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    
    gen_scaler = torch.cuda.amp.GradScaler()
    disc_scaler = torch.cuda.amp.GradScaler()

    for epcho in range(config.epchos):
        train(disc_h, disc_z, gen_h, gen_z, loader, opt_disc, opt_gen, F_loss_L1, F_loss_mse, gen_scaler, disc_scaler, epcho)

if __name__ == '__main__':
    main()