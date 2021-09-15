#based on https://www.youtube.com/watch?v=nkQHASviYac&list=PLhhyoLH6IjfwIp8bZnzX8QR30TRcHO8Va&index=12&t=16s
import torch
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import gradient_penalty, plot_to_tensorboard, generate_examples
from model import Discriminator, Generator
from math import log2
from tqdm import tqdm
from config import *

torch.backends.cudnn.benchmark = True # preformence benifit

def get_loader(image_size):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize([0.5 for _ in range(channels_img)], [0.5 for _ in range(channels_img)])
        ]
    )

    batch_size = batch_sizes[int(log2(image_size / 4))]
    dataset = datasets.ImageFolder(root=data, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    return loader, dataset

def train(disc, gen, loader, dataset, step, alpha, opt_disc, opt_gen, tensorboard_step, writer, scaler_gen, scaler_disc):
    loop = tqdm(loader, leave=True)

    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(device)
        cur_batch_size = real.shape[0]

        # Train Critic: max E[critic(real)] - E[critic(fake)] <=> min -E[critic(real)] + E[critic(fake)]
        noise = torch.randn(cur_batch_size, z_dim, 1, 1).to(device)

        with torch.cuda.amp.autocast():
            fake = gen(noise, alpha, step)
            disc_real = disc(real, alpha, step)
            disc_fake = disc(fake.detach(), alpha, step)

            gp = gradient_penalty(disc, real, fake, alpha, step, device=device)

            loss_disc = (-(torch.mean(disc_real) - torch.mean(disc_fake)) + lambada_gp * gp + (0.001 * torch.mean(disc_real ** 2)))

        opt_disc.zero_grad()
        scaler_disc.scale(loss_disc).backward()
        scaler_disc.step(opt_disc)
        scaler_disc.update()

        # Train Generator: max E[critic(gen_fake)] <=> min -E[critic(gen_fake)]
        with torch.cuda.amp.autocast():
            gen_fake = disc(fake, alpha, step)
            loss_gen = -torch.mean(gen_fake)
        
        opt_gen.zero_grad()
        scaler_gen.scale(loss_gen).backward()
        scaler_gen.step(opt_gen)
        scaler_gen.update()

        alpha += cur_batch_size / (len(dataset) * progressive_epoch[step]*0.5)
        alpha = min(alpha, 1)

        if batch_idx % 500 == 0:
            with torch.no_grad():
                fixed_fakes = gen(fixed_noise, alpha, step) * 0.5 + 0.5

            plot_to_tensorboard( writer, loss_disc.item(), loss_gen.item(), real.detach(), fixed_fakes.detach(), tensorboard_step)

            tensorboard_step += 1
        
        if save_model:
            torch.save(disc.state_dict(), f'checkpoint/Disc_checkpoint.pth')
            torch.save(disc.state_dict(), f'checkpoint/Gen_checkpoint.pth')

    return tensorboard_step, alpha

def main():
    gen = Generator(z_dim, in_channels, channels_img).to(device)
    disc = Discriminator(in_channels, channels_img).to(device)

    #optimizers
    opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.0, 0.99))
    opt_disc = optim.Adam(disc.parameters(), lr=lr, betas=(0.0, 0.99))
    scaler_disc = torch.cuda.amp.GradScaler()
    scaler_gen = torch.cuda.amp.GradScaler()

    #for tensorboard
    writer = SummaryWriter(f'logs/gan')

    if load_model:
        gen.load_state_dict(torch.load('checkpoint/Gen_checkpoint.pth'))
        disc.load_state_dict(torch.load('checkpoint/Disc_checkpoint.pth'))

    gen.train()
    disc.train()

    tensorboard_step = 0
    step = int(log2(start_train_at_img_size/4))

    for num_epochs in progressive_epoch[step:]:
        alpha=1e-5
        loader, dataset = get_loader(4*2**step)
        print(f'image size: {4*2**step}')

        for epoch in range(num_epochs):
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            tensorboard_step, alpha = train(disc, gen, loader, dataset, step, alpha, opt_disc, opt_gen, tensorboard_step, writer, scaler_gen, scaler_disc)
        step += 1

if __name__ == '__main__':
    main()