import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels=8, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))  # out_channels is no. of filters and this padding size provide with same convolution.
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))  # out_channels is no. of filters and this padding size provide with same convolution.
        self.fc1 = nn.Linear(in_features=16*7*7, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)  # flattening the tensor and maintaining the batch size
        x = self.fc1(x)
        return x




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

in_channels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 3

# dataloading
train_dataset = datasets.MNIST(root = 'dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model = CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        scores = model(data)
        loss = criterion(scores, targets)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

def accuracy(loader, model):
    if loader.dataset.train:
        print("checking train acc. ")
    else:
        print("checking test acc. ")


    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            #x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f"got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}")
    model.train()
def check_gpu_usage():
    if torch.cuda.is_available():
        print(f"Allocated GPU memory: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        print(f"Reserved GPU memory: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")
    else:
        print("CUDA is not available.")

def show_random_images_and_predictions(model, test_loader):
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images[:2].to(device), labels[:2].to(device)  # Get the first 2 images from the batch

    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

    model.train()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for idx, ax in enumerate(axes):
        ax.imshow(images[idx].cpu().squeeze(), cmap='gray')
        ax.set_title(f"Predicted: {preds[idx].item()}, Actual: {labels[idx].item()}")
        ax.axis('off')
    plt.show()


check_gpu_usage()

accuracy(train_loader, model)
accuracy(test_loader, model)

show_random_images_and_predictions(model, test_loader)