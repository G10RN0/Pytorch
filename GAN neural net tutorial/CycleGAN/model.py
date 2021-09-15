import torch
import torch.nn as nn
from torch.nn.modules import instancenorm

class Discriminator(nn.Module):
    def __init__(self, in_channels, features):
        super(Discriminator, self).__init__()

        self.L_relu = nn.LeakyReLU(0.2)

        self.conv_in = nn.Conv2d(in_channels, features, stride=2, kernel_size=4, padding=1, padding_mode='reflect')

        self.conv_1 = nn.Conv2d(features, features*2, stride=2, kernel_size=4, padding=1, bias=True, padding_mode='reflect')
        self.instanceNorm2d_1 = nn.InstanceNorm2d(features*2)

        self.conv_2 = nn.Conv2d(features*2, features*4, stride=2, kernel_size=4, padding=1, bias=True, padding_mode='reflect')
        self.instanceNorm2d_2 = nn.InstanceNorm2d(features*4)

        self.conv_3 = nn.Conv2d(features*4, features*8, stride=1, kernel_size=4, padding=1, bias=True, padding_mode='reflect')
        self.instanceNorm2d_3 = nn.InstanceNorm2d(features*8)

        self.conv_out = nn.Conv2d(features*8, 1, stride=1, kernel_size=4, padding=1, padding_mode='reflect')
    
    def forward(self, x):

        x = self.L_relu(self.conv_in(x))

        x = self.L_relu(self.instanceNorm2d_1(self.conv_1(x)))
        x = self.L_relu(self.instanceNorm2d_2(self.conv_2(x)))
        x = self.L_relu(self.instanceNorm2d_3(self.conv_3(x)))

        x = self.conv_out(x)

        return torch.sigmoid(x)

class Generator(nn.Module):
    def __init__(self, in_channels, features):
        super(Generator, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.identity = nn.Identity()
        
        
        self.conv_in = nn.Conv2d(in_channels, features, kernel_size=7, stride=1, padding=3, padding_mode='reflect')

        self.conv_1 = nn.Conv2d(features, features*2, kernel_size=3, stride=2, padding=1, padding_mode='reflect')
        self.instanceNorm2d_1 = nn.InstanceNorm2d(features*2)
        self.conv_2 = nn.Conv2d(features*2, features*4, kernel_size=3, stride=2, padding=1, padding_mode='reflect')
        self.instanceNorm2d_2 = nn.InstanceNorm2d(features*4)

        #9 razy powtarzamy bo to jest blok
        self.conv_chain_1 = nn.Conv2d(features*4, features*4, kernel_size=3, padding=1, padding_mode='reflect')
        self.instanceNorm2d_chain_1 = nn.InstanceNorm2d(features*4)
        #drugi ma Identity
        self.conv_chain_2 = nn.Conv2d(features*4, features*4, kernel_size=3, padding=1, padding_mode='reflect')
        self.instanceNorm2d_chain_2 = nn.InstanceNorm2d(features*4)

        self.T_conv_1 = nn.ConvTranspose2d(features*4, features*2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.instanceNorm2d_3 = nn.InstanceNorm2d(features*2)
        self.T_conv_2 = nn.ConvTranspose2d(features*2, features, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.instanceNorm2d_4 = nn.InstanceNorm2d(features)
        
        self.conv_out = nn.Conv2d(features, 3, kernel_size=7, stride=1, padding=3, padding_mode='reflect')

    def forward(self, x):
        
        x = self.relu(self.conv_in(x))

        x = self.relu(self.instanceNorm2d_1(self.conv_1(x)))
        x = self.relu(self.instanceNorm2d_2(self.conv_2(x)))
        for _ in range(9):
            x = self.relu(self.instanceNorm2d_chain_1(self.conv_chain_1(x)))
            x = self.identity(self.instanceNorm2d_chain_2(self.conv_chain_2(x)))

        x = self.relu(self.instanceNorm2d_3(self.T_conv_1(x)))
        x = self.relu(self.instanceNorm2d_4(self.T_conv_2(x)))

        x = self.conv_out(x)

        return torch.tanh(x)

def test():
    imgs = torch.randn(1, 3, 256, 256)
    gen = Generator(3, 64)
    dis = Discriminator(3, 64)
    img = gen(imgs)
    pred = dis(imgs)
    print(img.shape)
    print(pred.shape)

if __name__ == '__main__':
    test()