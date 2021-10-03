import os
import random

import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.datasets import ImageFolder
def getFirst(e):
    return e[0]

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, transformers):
        self.root = root
        self.transforms = transformers
        # load all image files, sorting them to
        # ensure that they are aligned
        self.classes = torch.tensor([0,1,2,3,4,5,6,7,8,9,10])
        self.imgs = []
        for index in range(11):
            newimgs = list(sorted(os.listdir("Data/food-11/"+root+"/"+str(index))))
            for i in newimgs:
                b = i.split('.')
                if (b[1] == "jpg") & (len(i) > 0):
                    self.imgs.append(["Data/food-11/"+root+"/"+str(index)+"/"+i,self.classes[index]])
        #self.imgs.sort(key=getFirst)



    def __getitem__(self, idx):
        # load images and masks
        ImgPath,ImgClass = self.imgs[idx]
        img = self.transforms(Image.open(ImgPath))
        return(img,ImgClass)



    def __len__(self):
        return len(self.imgs)


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

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3,6,(5,5))
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,(5,5))
        self.fc1 = nn.Linear(16*29*29,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,11)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,16*29*29)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train():

    transform = transforms.Compose(
        [transforms.Resize((128,128)), transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = CustomDataset("training",transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=2)

    testset = CustomDataset("validation",transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=True, num_workers=2)


    classes = (
        'Bread', 'Dairy', 'Dessert', 'Egg', 'Fried', 'Meat', 'Pasta', 'Rice',
        'Seafood', 'Soup',
        'Vegetable/Fruit')


    dataiter = iter(trainloader)
    images, labels = next(dataiter)
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

    dataiter2 = iter(testloader)
    images2, labels2 = next(dataiter2)
    outputs = net(images2)
    _, predicted = torch.max(outputs,1)
    s1 = "Pred:"+' '.join('%5s' % classes[predicted[j]] for j in range(16))
    s2 = "Actual:"+' '.join('%5s' % classes[labels2[j]] for j in range(16))
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(16)))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels2[j]] for j in range(16)))
    imshow(torchvision.utils.make_grid(images2),s1+"\n"+s2)
    score = 0
    for a in range(len(predicted)):
        if predicted[a]==labels2[a]:
            score+=1
    print("Accuracy = "+str(score/16))




def run():
    torch.multiprocessing.freeze_support()
    print('loop')

if __name__ == '__main__':
    run()
    train()
