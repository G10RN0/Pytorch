#aby zrozumiec obejrz to https://www.youtube.com/watch?v=1gQR24B3ISE&list=PLQVvvaa0QuDdeMyHEYc0gxFpYwHY2Qfdh&index=6

#aby sprawdzic jaki rozmiar ma fc1 to tezba to zrobic
'''
img = torch.randn(64, 64).view(-1, 1, 64, 64)

conv1 = nn.Conv2d(1, 32, 5)
pool = nn.MaxPool2d(2,2)
conv2 = nn.Conv2d(32, 64, 5)
conv3 = nn.Conv2d(64, 128, 5)

x = conv1(img)
x = pool(x)
x = conv2(x)
x = pool(x)
x = conv3(x)
x = pool(x)
print(x.shape)
'''
#torch.Size([1, 128, 4, 4])


from typing import NewType
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        
        x = torch.randn(64, 64).view(-1, 1, 64, 64)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 2)

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        if self._to_linear is None:
            #zdobywamy wymiary
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        
        return x
    def forward(self, x):
        x = self.convs(x)

        #flatten
        x = x.view(-1 ,self._to_linear)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)
        

net = Net()
print(net)

optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_f = nn.MSELoss()

epchos = 10
batch_size = 64

print('loading train data...')
train_data = np.load('convlution_neural_net/train_data.npy', allow_pickle=True)
print('loading test data...')
test_data = np.load('convlution_neural_net/test_data.npy', allow_pickle=True)

print('formating train data...')
train_x = torch.Tensor([i[0] for i in train_data]).view(-1, 64, 64)
train_x = train_x/255.0
train_y = torch.Tensor([i[1] for i in train_data])

print('formating test data...')
test_x = torch.Tensor([i[0] for i in test_data]).view(-1, 64, 64)
test_x = test_x/255.0
test_y = torch.Tensor([i[1] for i in test_data])

def validation(net ,x , y):
    correct = 0
    total = 0
    x = x
    #validation
    #robi to aby nasz net sie nie uczyl
    with torch.no_grad():
        for i in range(len(x)):
            real_class = torch.argmax(y)
            net_out = net(test_x[i].view(-1, 1, 64, 64))[0]

            predicted_class = torch.argmax(net_out)
            if predicted_class == real_class:
                correct +=1
            total += 1

    return correct/total

print('starting trianing...')

for epcho in range(epchos):
    for i in tqdm(range(0, len(train_x), batch_size)):

        batch_x = train_x[i:i+batch_size].view(-1, 1, 64, 64)
        batch_y = train_y[i:i+batch_size]

        net.zero_grad()
        outputs = net(batch_x)
        loss = loss_f(outputs, batch_y)
        loss.backward()
        optimizer.step()

    accuracy = validation(net, test_x, test_y)
    print(f'Epoch: {epcho+1}, loss: {loss}, validation: {accuracy}')