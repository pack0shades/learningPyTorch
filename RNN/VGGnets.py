import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
learning_rate = 0.001
num_epochs = 10
batch_size = 32

# VGG requires input to be 224x224 size
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    #transforms.Normalize((0.5), (0.5))
])
train_data = datsets.MNIST(root='dataset/MNIST', train=True, transform=transform, download=True)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_data = datsets.MNIST(root='dataset/MNIST', train=False, transform=transform, download=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
VGG = {
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512,'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        }


class VGGnet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(VGGnet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.conv_layers = self.create_convlayers(VGG['VGG11'])

        self.fcs = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x

    def create_convlayers(self, network):
        layers = []
        in_channels = self.in_channels

        for x in network:
            if type(x) == int:
                out_channels = x

                layers += [nn.Conv2d(in_channels=in_channels,out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),nn.BatchNorm2d(x),
                           nn.ReLU()]

                in_channels = x

            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return nn.Sequential(*layers)

criterion = nn.CrossEntropyLoss()

model = VGGnet(in_channels=3, num_classes=10).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# training
starttime = time.time()
for epoch in range(num_epochs):
    for batch_idx,(data, targets) in enumerate(train_loader):
        data = data.to(DEVICE)
        targets = targets.to(DEVICE)

        pred = model(data)
        loss = criterion(pred, targets)

        optimize.zero_grad()
        loss.backward()

        optimizer.step()

    print(f'epoch({epoch+1}/{num_epochs})\nLoss : {loss.item():.4f}')

def accuracy(loader, model):
    if loader.dataset.train:
        print("train acc. = ")
    else:
        print("test acc. = ")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            pred = model(x)
            i, prediction = pred.max(1)
            num_correct += (prediction == y).sum()
            num_samples += prediction.size(0)

            print(f'{float(num_correct) / float(num_samples) * 100:.2f}\n')
            print(f'ye kya hai {i}')
    model.train()

    accuracy(loader=train_loader, model=model)
    accuracy(loader=test_loader, model=model)