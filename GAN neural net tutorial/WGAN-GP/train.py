import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import os
from model import Critic, Generator, initialize_weights
from utils import gradient_penalty

#gpu
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

#variables
img_size = 64
channels = 3
batch_size = 64
z_dim = 100
learning_rate = 1e-4
epchos = 1000
features_disc = 64
features_gen = 64
critc_iterations = 5
lambda_GP = 10

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
critic = Critic(channels, features_disc).to(device)
initialize_weights(critic)

gen = Generator(z_dim, features_gen).to(device)
initialize_weights(gen)

#optimizers
opt_critic = optim.Adam(critic.parameters(), lr=learning_rate, betas=(0.0, 0.9))
opt_gen = optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.0, 0.9))
loss_F = nn.BCELoss()

#test noise
test_noise = torch.randn(32, 100, 1, 1).to(device)

#setup tensrboard
log_real = SummaryWriter(f'logs/real')
log_fake = SummaryWriter(f'logs/fake')

#train preperations
critic.train()
gen.train()

step = 0
for epcho in range(epchos):
    print(f'Epchos: {epcho}/{epchos}')
    for batch_idx, (real, _) in enumerate(tqdm(loader)):
        real = real.to(device)
        cur_batch_size = real.shape[0]
        
        #critic loss
        # Train Critic: max E[critic(real)] - E[critic(fake)]
        # equivalent to minimizing the negative of that
        for _ in range(critc_iterations):
            noise = torch.randn(cur_batch_size, z_dim, 1, 1).to(device)
            fake = gen(noise)

            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)

            gp = gradient_penalty(critic, real, fake, device=device)

            critic_loss = (
                -(torch.mean(critic_real) - torch.mean(critic_fake)) + lambda_GP * gp
                )

            critic.zero_grad()
            critic_loss.backward(retain_graph=True)
            opt_critic.step()

        #generator loss: min -E(critic(gen_fake))
        output = critic(fake).reshape(-1)
        gen_loss = -(torch.mean(output))

        gen.zero_grad()
        gen_loss.backward()
        opt_gen.step()

        #updating tensrboard and displaying loss
        if batch_idx % 1500 == 0 and batch_idx > 0:
            print(f' Critic loss: {critic_loss}, Generator loss: {gen_loss}')
            with torch.no_grad():
                fake = gen(test_noise)

                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                log_real.add_image('Real', img_grid_real, global_step=step)
                log_fake.add_image('Fake', img_grid_fake, global_step=step)

                save_image(img_grid_fake, f'generated_images/step{step}.jpg')

                torch.save(critic.state_dict(), f'models/Critic_epcho{epcho}_step{step}.pth')
                torch.save(gen.state_dict(), f'models/Generator_epcho{epcho}_step{step}.pth')

                step += 1