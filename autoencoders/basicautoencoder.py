import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from helper_autoencoder import Dataloader_catdog
from helper_autoencoder import train_autoencoder
from helper_autoencoder import plot_generated_images
DEVICE = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', DEVICE)

num_epochs = 100
batch_size = 32
learning_rate = 0.0001
RANDOM_SEED = 123


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # (3, 64, 64) -> (16, 32, 32)
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # (16, 32, 32) -> (32, 16, 16)
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)  # (32, 16, 16) -> (64, 10, 10)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),  # (64, 10, 10) -> (32, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # (32, 16, 16) -> (16, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),  # (16, 32, 32) -> (3, 64, 64)
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = ConvAutoencoder()
criterion = nn.MSELoss()
model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



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


# Training

results = train_autoencoder(num_epochs=num_epochs,model= model, optimizer=optimizer, device=DEVICE, train_loader=train_loader, loss_fn=None, logging_interval=100, skip_epoch_stats=False,
                             save_model=None)

print(results)

minibatch_losses = results['train_loss_per_batch']
custom_label = ''
averaging_iterations=100
iter_per_epoch = len(minibatch_losses) // num_epochs

plt.figure()
ax1 = plt.subplot(1, 1, 1)
ax1.plot(range(len(minibatch_losses)),
             (minibatch_losses), label=f'Minibatch Loss{custom_label}')
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Loss')

if len(minibatch_losses) < 1000:
        num_losses = len(minibatch_losses) // 2
else:
        num_losses = 1000

ax1.set_ylim([
        0, np.max(minibatch_losses[num_losses:])*1.5
        ])

ax1.plot(np.convolve(minibatch_losses,
                         np.ones(averaging_iterations,)/averaging_iterations,
                         mode='valid'),
             label=f'Running Average{custom_label}')
ax1.legend()

###################
# Set scond x-axis
ax2 = ax1.twiny()
newlabel = list(range(num_epochs+1))

newpos = [e*iter_per_epoch for e in newlabel]

ax2.set_xticks(newpos[::10])
ax2.set_xticklabels(newlabel[::10])

ax2.xaxis.set_ticks_position('bottom')
ax2.xaxis.set_label_position('bottom')
ax2.spines['bottom'].set_position(('outward', 45))
ax2.set_xlabel('Epochs')
ax2.set_xlim(ax1.get_xlim())
###################

plt.tight_layout()

plot_generated_images(data_loader=test_loader, model=model, device=DEVICE)
plt.show()
