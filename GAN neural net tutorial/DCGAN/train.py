import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import os
from model import Discriminator, Generator, initialize_weights

#gpu
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

#variables
img_size = 64
channels = 3
batch_size = 16
z_dim = 100
learning_rate = 2e-4
epchos = 1000
features_disc = 64
features_gen = 64

#transforms
transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((img_size, img_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5 for _ in range(channels)], [0.5 for _ in range(channels)])
    ]
)

#loads dataset
dataset = datasets.ImageFolder(root='dataset', transform=transforms)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#loads models
disc = Discriminator(channels, features_disc).to(device)
disc.load_state_dict(torch.load('models/Discriminator_epcho478_step3825.pth'))

gen = Generator(z_dim, features_gen).to(device)
gen.load_state_dict(torch.load('models/Generator_epcho478_step3825.pth'))

#optimizers
opt_disc = optim.Adam(disc.parameters(), lr=learning_rate, betas=(0.5, 0.999))
opt_gen = optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.5, 0.999))
loss_F = nn.BCELoss()

#test noise
test_noise = torch.randn(32, 100, 1, 1).to(device)

#setup tensrboard
log_real = SummaryWriter(f'logs/real')
log_fake = SummaryWriter(f'logs/fake')

#train preperations
disc.train()
gen.train()

step = 3825

for epcho in range(epchos):
    epcho = epcho+478
    print(f'Epchos: {epcho}/{epchos}')
    for batch_idx, (real, _) in enumerate(tqdm(loader)):
        real = real.to(device)
        noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
        fake = gen(noise)

        #discriminator loss
        disc_real = disc(real).reshape(-1)
        disc_real_loss = loss_F(disc_real, torch.ones_like(disc_real))

        disc_fake = disc(fake.detach()).reshape(-1)
        disc_fake_loss = loss_F(disc_fake, torch.zeros_like(disc_fake))

        disc_loss = (disc_real_loss + disc_fake_loss) / 2

        disc.zero_grad()
        disc_loss.backward()
        opt_disc.step()

        #generator loss
        output = disc(fake).reshape(-1)
        gen_loss = loss_F(output, torch.ones_like(output))

        gen.zero_grad()
        gen_loss.backward()
        opt_gen.step()

        #updating tensrboard and displaying loss
        if batch_idx % 100 == 0:
            print(f' Discrimanator loss: {disc_loss}, Generator loss: {gen_loss}')
            with torch.no_grad():
                fake = gen(test_noise)

                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                log_real.add_image('Real', img_grid_real, global_step=step)
                log_fake.add_image('Fake', img_grid_fake, global_step=step)

                np.save(os.path.join('generated_images', f'epcho{epcho}_step{step}.npy'), fake[0].cpu().numpy())

                torch.save(disc.state_dict(), f'models/Discriminator_epcho{epcho}_step{step}.pth')
                torch.save(gen.state_dict(), f'models/Generator_epcho{epcho}_step{step}.pth')

                step += 1