import random

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

def imshow(img,s=""):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if s and not '\n' in s:
        s = ' '.join(s.split())
        p = s.find(' ',int(len(s)/2))
        s = s[:p]+"\n"+s[p+1:]
    plt.text(0,-20,s)
    plt.show()



import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,11)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train():

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((32,32)),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.ImageFolder('data/food-11/training', transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.ImageFolder('data/food-11/training', transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                             shuffle=False, num_workers=2)

    classes = (
        'Bread', 'Dairy product', 'Dessert', 'Egg', 'Fried food', 'Meat', 'Noodles/Pasta', 'Rice', 'Seafood', 'Soup',
        'Vegetable/Fruit')

    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    s = ' '.join('%5s' % classes[labels[j]] for j in range(16))
    print(s)
    imshow(torchvision.utils.make_grid(images),s)
    print(1)

    net = Net()
    #import torchvision.models as models
    #net = models.resnet18(pretrained=True)
    #net.fc = nn.Linear(512,10)
    #import pdb;pdb.set_trace()
    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if(i%500==0):
                print(epoch,i,running_loss/(i+1))

    dataiter = iter(testloader)
    images, labels = dataiter.next()
    outputs = net(images)
    _, predicted = torch.max(outputs,1)
    s1 = "Pred:"+' '.join('%5s' % classes[predicted[j]] for j in range(16))
    s2 = "Actual:"+' '.join('%5s' % classes[labels[j]] for j in range(16))
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(16)))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(16)))
    imshow(torchvision.utils.make_grid(images),s1+"\n"+s2)



def run():
    torch.multiprocessing.freeze_support()
    print('loop')

if __name__ == '__main__':
    run()
    train()
