import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input, hidden):
        super(Discriminator, self).__init__()

        self.L_relu = nn.LeakyReLU(0.2)

        self.conv_in = nn.Conv2d(input, hidden, kernel_size=4,stride=2,padding=1)#32x32

        self.conv_1 = nn.Conv2d(hidden, hidden*2, kernel_size=4,stride=2,padding=1)#16x16
        self.batch_norm_2d_1 = nn.BatchNorm2d(hidden*2)
        self.conv_2 = nn.Conv2d(hidden*2, hidden*4, kernel_size=4,stride=2,padding=1)#8x8
        self.batch_norm_2d_2 = nn.BatchNorm2d(hidden*4)
        self.conv_3 = nn.Conv2d(hidden*4, hidden*8, kernel_size=4,stride=2,padding=1)#4x4
        self.batch_norm_2d_3 = nn.BatchNorm2d(hidden*8)

        self.conv_out = nn.Conv2d(hidden*8, 1, kernel_size=4,stride=2,padding=0)#1x1
    
    def forward(self, x):
    
        x = self.L_relu(self.conv_in(x))

        x = self.L_relu(self.batch_norm_2d_1(self.conv_1(x)))
        x = self.L_relu(self.batch_norm_2d_2(self.conv_2(x)))
        x = self.L_relu(self.batch_norm_2d_3(self.conv_3(x)))

        x = self.conv_out(x)

        return torch.sigmoid(x)

class Generator(nn.Module):
    def __init__(self, z_dim, hidden):
        super(Generator, self).__init__()
        
        self.L_relu = nn.LeakyReLU(0.2)

        self.Tconv_in = nn.ConvTranspose2d(z_dim, hidden*16, kernel_size=4, stride=2, padding=0)#4x4

        self.Tconv_1 = nn.ConvTranspose2d(hidden*16, hidden*8, kernel_size=4, stride=2, padding=1)#8x8
        self.batch_norm_2d_1 = nn.BatchNorm2d(hidden*8)
        self.Tconv_2 = nn.ConvTranspose2d(hidden*8, hidden*4, kernel_size=4, stride=2, padding=1)#16x16
        self.batch_norm_2d_2 = nn.BatchNorm2d(hidden*4)
        self.Tconv_3 = nn.ConvTranspose2d(hidden*4, hidden*2, kernel_size=4, stride=2, padding=1)#32x32
        self.batch_norm_2d_3 = nn.BatchNorm2d(hidden*2)

        self.Tconv_out = nn.ConvTranspose2d(hidden*2, 3, kernel_size=4, stride=2, padding=1)#64x64

    def forward(self, x):

        x = self.L_relu(self.Tconv_in(x))

        x = self.L_relu(self.batch_norm_2d_1(self.Tconv_1(x)))
        x = self.L_relu(self.batch_norm_2d_2(self.Tconv_2(x)))
        x = self.L_relu(self.batch_norm_2d_3(self.Tconv_3(x)))

        x = self.Tconv_out(x)

        return torch.tanh(x)
    
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.2)

def test():

    img = torch.randn(1, 3, 64, 64)
    disc = Discriminator(3, 64)
    initialize_weights(disc)
    assert disc(img).shape == (1, 1, 1, 1)

    noise = torch.randn(1, 100, 1, 1)
    gen = Generator(100, 64)
    initialize_weights(gen)
    assert gen(noise).shape == (1, 3, 64, 64)

test()