import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import tqdm
from torchvision import transforms, datasets

train = datasets.MNIST("", train=True, download=True,
        transform=transforms.Compose([transforms.ToTensor()]))

test = datasets.MNIST("", train=False, download=True,
        transform=transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=64, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm1 = nn.LSTM(28, 32, 1, batch_first=True)
        self.lstm2 = nn.LSTM(32, 64, 1, batch_first=True)
        self.lstm3 = nn.LSTM(64, 128, 1, batch_first=True)

        self.fc1 = nn.Linear(28*128, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        h1 = torch.zeros(1, x.size(0), 32, dtype=torch.float32)
        c1 = torch.zeros(1, x.size(0), 32, dtype=torch.float32)

        h2 = torch.zeros(1, x.size(0), 64, dtype=torch.float32)
        c2 = torch.zeros(1, x.size(0), 64, dtype=torch.float32) 

        h3 = torch.zeros(1, x.size(0), 128, dtype=torch.float32)
        c3 = torch.zeros(1, x.size(0), 128, dtype=torch.float32)  

        h1, c1 = self.lstm1(x, (h1, c1))
        h2, c2 = self.lstm2(h1, (h2, c2))
        x, _ = self.lstm3(h2, (h3, c3))

        #flatten
        x = x.reshape(-1, 28*128)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.softmax(x, dim=1)

net = Net()

print(net)

optimazier = optim.Adam(net.parameters(), lr=0.001)

epochs = 10

def validation(net):
        correct = 0
        total = 0
        #validation
        #robi to aby nasz net sie nie uczyl
        with torch.no_grad():
                for data in testset:
                        #nasz data i labels
                        x, y  = data

                        #wkladamy zdjecia do net
                        output = net(x.view(-1, 28, 28))

                        #nie wiem to co robi. mysle ze poprostu zabiermy wynik net
                        for idx, i in enumerate(output):
                                #jesli wynik net jest dobry to dodajemy 1 do correct
                                if torch.argmax(i) == y[idx]:
                                        correct += 1
                                total += 1
        
        #zaokraglamy liczbe do 3 liczb inaczej do jednej liczby po przecinku
        x = round(correct/total, 3)
        return x

#training process
for epoch in range(epochs):
        #data to jest batch zdjec i lables
        z = 0
        for data in trainset:
                #nasz data i labels
                x, y = data

                #zero_grad nie wiem co to robi
                net.zero_grad()
                #wkladamy zdjecia do net
                output = net(x.view(-1, 28, 28))
                #przeczytaj jaki loss do jakich rzeczy
                loss = F.nll_loss(output, y)
                #we backprogade loss
                loss.backward()
                #poprawiamy weights
                optimazier.step()
                z += 1
                print(f'step: {z} from {int(len(trainset))}')
        
        accuracy = validation(net)
        print(f'Epoch: {epoch+1}, loss: {loss}, validation: {accuracy}')
