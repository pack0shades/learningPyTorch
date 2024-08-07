import torch
import torch.nn as nn
import torch.nn.functional as F
from helper_autoencoder import Dataloader_catdog
from tqdm import tqdm
from helper_autoencoder import plot_generated_images
import matplotlib as plt
device = "cuda" if torch.cuda.is_available() else 'cpu'
print(device)

learning_rate = 0.001
epochs = 10
batch_size = 64
latent_dim = 2
train_loader, test_loader = Dataloader_catdog(batch_size=batch_size, data_transform=None)

# checking dataset
print('Training Set:\n')
for images, labels in train_loader:
    print('Image batch dimensions:', images.size())
    print('Image label dimensions:', labels.size())
    print(labels[:10])
    break

# Checking the dataset
print('\nTesting Set:')
for images, labels in test_loader:
    print('Image batch dimensions:', images.size())
    print('Image label dimensions:', labels.size())
    print(labels[:10])
    break


class samplling(nn.Module):

    def forward(self, mean, log_var):
        batch = mean.size(0)
        dim = mean.size(1)
        epsilon = torch.randn(batch, dim, device=mean.device)
        return mean + torch.exp(0.5 * log_var) * epsilon


class Encoder(nn.Module):
    def __init__(self, latent=latent_dim):
        super(Encoder, self).__init__()
        self.conv_1 = nn.Conv2d(3, 16, 3, 2, 1)
        self.conv_2 = nn.Conv2d(16, 32, 3, 2, 1)
        self.conv_3 = nn.Conv2d(32, 64, 3,2, 1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64*8*8, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc_mean = nn.Linear(16, latent)
        self.fc_log_var = nn.Linear(16, latent)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = F.relu(self.conv_3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        mean = self.fc_mean(x)
        log_var = self.fc_log_var(x)
        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, latent):
        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(latent, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 64*8*8)
        self.conv_1 = nn.ConvTranspose2d(64, 32, 3, 2, 1, 1)
        self.conv_2 = nn.ConvTranspose2d(32, 16, 3, 2, 1, 1)
        self.conv_3 = nn.ConvTranspose2d(16, 3, 3, 2, 1, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, 64, 8, 8)
        x = F.relu(self.conv_1(x))
        x = self.conv_2(x)
        x = torch.sigmoid(self.conv_3(x))
        return x


class VAE(nn.Module):
    def __init__(self, encode, decode):
        super(VAE, self).__init__()

        self.encoder = encode
        self.decoder = decode
        self.samplling = samplling()

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.samplling(mean, log_var)
        reconst = self.decoder(z)
        return reconst, mean, log_var

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x.view(-1, 64*64*3), x.view(-1, 64*64*3), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD


encoder = Encoder(latent_dim).to(device)
decoder = Decoder(latent_dim).to(device)
vae = VAE(encoder, decoder).to(device)
optimizer = torch.optim.Adam(params=vae.parameters(), lr=learning_rate, weight_decay=0.01)


def train(model, optimizer, epochs, loader):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(tqdm(loader)):
            data = data.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(data)
            loss = model.loss_function(recon, data, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(loader.dataset)}")


def test(model, loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(loader):
            data = data.to(device)
            recon, mu, logvar = model(data)
            loss = model.loss_function(recon, data, mu, logvar)
            total_loss += loss.item()
    print(f"Test Loss: {total_loss / len(loader.dataset)}")


train(vae, optimizer, epochs, train_loader)
test(vae, test_loader)

plot_generated_images(test_loader, model=vae, device=device, modeltype="VAE")
