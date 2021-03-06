# Baloons assignment
import json
import os
from PIL import Image
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms



class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, transformers):
        self.root = root
        self.transforms = transformers
        self.imgs = []
        self.labels = {}
        filepath = open("Data/Balloons/balloon/"+root+"/via_region_data.json")
        file = json.load(filepath) # returns dict of json info
        jsondata = list(file.values())
        imgdata = [img for img in jsondata if img['regions']]
        for img in imgdata:
            path = "Data/Balloons/balloon/"+root+img["filename"]
            self.imgs.append(path)
            # array per image with separate dict for each region
            polygons = [region['shape_attributes'] for region in img['regions'].values()]
            self.labels[path]=polygons
        print(self.labels.keys())
        exit()


        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }



    def __getitem__(self, idx):
        # load images and masks
        ImgPath = self.imgs[idx]
        all_labels = self.labels[ImgPath]
        img = self.transforms(Image.open(ImgPath,"r"))
        return(img)



    def __len__(self):
        return len(self.imgs)

### Example code ###

class NaiveResnet(nn.Module):
    def __init__(self, C_in):
        print("NaiveResnet init")
        super(NaiveResnet, self).__init__()
        self.tower_one = nn.Conv2d(C_in, 3, (1, 1))
        self.tower_two = nn.Conv2d(3, 3, (3, 3), padding=1)
        self.tower_three = nn.Conv2d(3, 3, (5, 5), padding=2)

    def forward(self, x):
        print("NaiveResnet forward")
        stream = self.tower_one(x).clamp(min=0)
        stream = self.tower_two(stream).clamp(min=0)
        stream = self.tower_three(stream).clamp(min=0)

        output = x.add(stream)  # stream + x #stream.add(x)#.add(stream)
        return output


class NaiveFireblock(nn.Module):
    def __init__(self, C_in):
        print("Naivefireblock init")
        super(NaiveFireblock, self).__init__()
        self.squeeze = nn.Conv2d(C_in, 3, (1, 1))
        self.tower_one = nn.Conv2d(3, 3, (1, 1))
        self.tower_two = nn.Conv2d(3, 3, (3, 3), padding=1)

    def forward(self, x):
        print("NaiveResnet forward")
        squeeze = self.squeeze(x).clamp(min=0)
        tower_one_stream = self.tower_one(squeeze).clamp(min=0)
        tower_two_stream = self.tower_two(squeeze).clamp(min=0)
        output = torch.cat([tower_one_stream, tower_two_stream],
                           dim=1)  # TODO: Check that I got the concatenation dimention right here
        return output


class NaiveInception(nn.Module):
    def __init__(self, C_in, ):
        print("NaiveInception init")
        super(NaiveInception, self).__init__()

        self.tower_one_1 = nn.Conv2d(C_in, 6, (3, 3), padding=1)
        self.tower_one_2 = nn.Conv2d(6, 6, (1, 1))

        self.tower_two_1 = nn.Conv2d(C_in, 6, (3, 3), padding=1)
        self.tower_two_2 = nn.Conv2d(6, 6, (1, 1))

        self.tower_three_1 = nn.Conv2d(C_in, 6, (3, 3), padding=1)
        self.tower_three_2 = nn.Conv2d(6, 6, (1, 1))

    def forward(self, x):
        print("NaiveResnet forward")
        tower_one_stream = self.tower_one_1(x).clamp(min=0)
        tower_one_stream = self.tower_one_2(tower_one_stream).clamp(min=0)

        tower_two_stream = self.tower_two_1(x).clamp(min=0)
        tower_two_stream = self.tower_two_2(tower_two_stream).clamp(min=0)

        tower_three_stream = self.tower_three_1(x).clamp(min=0)
        tower_three_stream = self.tower_three_2(tower_three_stream).clamp(min=0)

        output = torch.cat([tower_one_stream, tower_two_stream, tower_three_stream], dim=1)
        return output


class Net(nn.Module):
    def __init__(self):
        print("net init")
        super(Net, self).__init__()
        # self.res = NaiveResnet(3)
        self.res = NaiveInception(3)
        # self.conv = nn.Conv2d(3,6,(3,3),padding=1)

        # self.fc_1 = nn.Linear(3 * 32 * 32, 64) # You need to calculate the correct input to fc_1
        self.fc_1 = nn.Linear((6 + 6 + 6) * 32 * 32, 64)
        self.fc_2 = nn.Linear(64, 64)
        self.fc_3 = nn.Linear(64, 10)

    def forward(self, x):
        print("net forward")
        stream = self.res(x)
        # stream = self.conv(x)
        stream = torch.flatten(stream, start_dim=1)

        stream = self.fc_1(stream).clamp(min=0)
        stream = self.fc_2(stream).clamp(min=0)
        stream = self.fc_3(stream)

        output = F.log_softmax(stream, dim=1)
        return output


def train(model, device, train_loader, optimizer, epoch):
    print("train")
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 1000 == 0:
            print('Train Epoch: {} ({:.0f}%)\tLoss: {:.6f}'.format(
                epoch, 100. * batch_idx / len(train_loader), loss.item()))


def test1(model, device, test_loader):
    print("test")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Parameters
    data_dir = '/tmp/'

    BATCH_SIZE = 16
    EPOCHS = 5
    USE_CUDA = True and torch.cuda.is_available()

    device = torch.device('cuda' if USE_CUDA else 'cpu')

    data_loader_args = {
        'batch_size': BATCH_SIZE,
    }

    if USE_CUDA:
        data_loader_args.update({
            'num_workers': 1,
            'pin_memory': True,
            'shuffle': True
        })

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Import / download CIFAR dataset
    #trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=data_transform)
    #testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=data_transform)

    train_loader = torch.utils.data.DataLoader(trainset, **data_loader_args)
    test_loader = torch.utils.data.DataLoader(testset, **data_loader_args)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(1, EPOCHS + 1):
        train(model, device, train_loader, optimizer, epoch)
        test1(model, device, test_loader)


if __name__ == '__main__':
    print("dataloader")
    trainset = CustomDataset("train", None)
    testset = CustomDataset("val", None)
    print("main")

    main()
