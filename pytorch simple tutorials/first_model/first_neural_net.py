import torch
from torch.autograd import backward
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import torchvision
from torchvision import transforms, datasets

train = datasets.MNIST("", train=True, download=True,
        transform=transforms.Compose([transforms.ToTensor()]))

test = datasets.MNIST("", train=False, download=True,
        transform=transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

class Net(nn.Module):
        def __init__(self):
                super().__init__()

                #input
                self.fc1 = nn.Linear(28*28, 64)
                
                #hidden layer
                self.fc2 = nn.Linear(64, 64)
                self.fc3 = nn.Linear(64, 64)

                #output
                self.fc4 = nn.Linear(64, 10)
        
        #jaka opcje wybieramy jak ma przebiegac informacje(feed-forward), te ta funkcja musi miec specyficzna nazwe
        def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = F.relu(self.fc3(x))
                x = self.fc4(x)

                return F.log_softmax(x, dim=1)


net = Net()
print(net)

'''
X = torch.rand((28, 28))
#-1 symbolizuje ze to jest dowolna wilekosc
X = X.view(-1, 28*28)

output = net(X)
print(output)
'''

#optimazier(co ma byc opit)
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
                        output = net(x.view(-1, 28*28))

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
        for data in trainset:
                #nasz data i labels
                x, y = data

                #zero_grad nie wiem co to robi
                net.zero_grad()
                #wkladamy zdjecia do net
                output = net(x.view(-1, 28*28))
                #przeczytaj jaki loss do jakich rzeczy
                loss = F.nll_loss(output, y)
                #we backprogade loss
                loss.backward()
                #poprawiamy weights
                optimazier.step()
        
        accuracy = validation(net)
        print(f'Epoch: {epoch+1}, loss: {loss}, validation: {accuracy}')