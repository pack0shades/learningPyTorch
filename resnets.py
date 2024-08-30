## ResNet without using dropout with CIFAR10 datset accuracy was around 75%




import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torchsummary as summary
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
expansion = 4
num_epochs = 100
learning_rate = 2e-3

class ResNetblock(nn.Module):
    def __init__(self, in_channels, intermediate_channels, identity_downsample=None,stride=1):
        super(ResNetblock, self).__init__()
        self.expansion = expansion
        self.stride = stride
        self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(intermediate_channels, intermediate_channels*self.expansion, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(intermediate_channels*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.identity_downsample = identity_downsample

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        if self.identity_downsample is not None:
            residual = self.identity_downsample(residual)
        x += residual
        x = self.relu(x)
        return x
class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*4, num_classes)
        
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return x
    
    def _make_layer(self, block, intermediate_channels, num_residual_blocks, stride):
        identity_downsample = None
        layers = []
        
        if stride!= 1 or self.in_channels != intermediate_channels*4:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, intermediate_channels*4, kernel_size=1, stride=stride), nn.BatchNorm2d(intermediate_channels*4))
            
        layers.append(block(self.in_channels, intermediate_channels, identity_downsample,stride))
        self.in_channels = intermediate_channels * 4

        for i in range(num_residual_blocks-1):
            layers.append(block(self.in_channels, intermediate_channels))
            #self.in_channels = intermediate_channels * 4
        return nn.Sequential(*layers)


def resnet50(image_channels=3, num_classes=1000):
    return ResNet(ResNetblock, [3, 4, 6, 3], image_channels, num_classes)

def test():
    net = resnet50()
    x = torch.randn(2, 3, 224, 224)
    y = net(x).to(device)
    print(y.shape)
    
    
    

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, num_workers=4)
val_loader = DataLoader(dataset=val_dataset, batch_size=128, shuffle=False, num_workers=4)



criterion = nn.CrossEntropyLoss()
model = ResNet(ResNetblock, [3, 4, 6, 3], image_channels=3, num_classes=10).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Function to calculate accuracy
def accuracy(loader, model):
    if loader.dataset.train:
        print("Train Accuracy: ", end="")
    else:
        print("Test Accuracy: ", end="")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            _, prediction = pred.max(1)
            num_correct += (prediction == y).sum().item()
            num_samples += prediction.size(0)

        print(f'{float(num_correct) / float(num_samples) * 100:.2f}%\n')
    model.train()

starttime = time.time()
for epoch in range(num_epochs):
    epoch_starttime = time.time()
    running_loss = 0.0
    
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)):
        data = data.to(device)
        targets = targets.to(device)

        pred = model(data)
        loss = criterion(pred, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        time_elapsed = (time.time() - starttime) / 60
        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}, Time Elapsed: {time_elapsed:.2f} mins')

    avg_loss = running_loss / len(train_loader)
    epoch_runtime = (time.time() - epoch_starttime) / 60
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Runtime: {epoch_runtime:.2f} mins')

    accuracy(loader=train_loader, model=model)
    accuracy(loader=val_loader, model=model)
