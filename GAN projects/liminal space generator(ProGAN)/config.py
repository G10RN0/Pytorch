#based on https://www.youtube.com/watch?v=nkQHASviYac&list=PLhhyoLH6IjfwIp8bZnzX8QR30TRcHO8Va&index=12&t=16s
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

start_train_at_img_size = 4

data = 'dataset'
checkpoint_gen = "generator.pth"
checkpoint_disc = "critic.pth"

save_model = True
load_model = False

lr = 1e-3
batch_sizes = [32, 32, 32, 16, 16, 16, 16, 8, 4]

channels_img = 3
z_dim = 256  # should be 512 in original paper
in_channels = 256  # should be 512 in original paper

critic_iterations = 1
lambada_gp = 10
progressive_epoch = [30] * len(batch_sizes)
fixed_noise = torch.randn(8, z_dim, 1, 1).to(device)
num_workers = 4