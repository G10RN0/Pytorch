import torch
from model import Discriminator, Generator, initialize_weights
import matplotlib.pyplot as plt

noise = torch.randn(32, 100, 1, 1)

gen = Generator(100, 64)
gen.load_state_dict(torch.load('models/Generator_epcho147_step1176.pth'))

initialize_weights(gen)

fake = gen(noise)

for img in fake:
    img = (img.detach().numpy().reshape(64, 64, 3)*255).astype('uint8')

    imgplot = plt.imshow(img)
    plt.show()
