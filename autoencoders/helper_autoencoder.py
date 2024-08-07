import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import torch.nn.functional as F
import matplotlib.colors as mcolors
DEVICE = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', DEVICE)



def Dataloader_catdog(batch_size, data_transform=None):

    if data_transform is None:
        data_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])
    else: data_transform = data_transform

    train_dataset = datasets.ImageFolder(root='dataset/dog vs cat/dataset/training_set', transform=data_transform)
    test_dataset = datasets.ImageFolder(root='dataset/dog vs cat/dataset/test_set', transform=data_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    return train_loader,test_loader

def compute_epoch_loss_autoencoder(model, data_loader, loss_fn, device):
    model.eval()
    curr_loss, num_examples = 0., 0
    with torch.no_grad():
        for features, _ in data_loader:
            features = features.to(device)
            logits = model(features)
            loss = loss_fn(logits, features, reduction='sum')
            num_examples += features.size(0)
            curr_loss += loss

        curr_loss = curr_loss / num_examples
        return curr_loss


def train_autoencoder(num_epochs, model, optimizer, device,
                             train_loader, loss_fn=None,
                             logging_interval=100,
                             skip_epoch_stats=False,
                             save_model=None):

        log_dict = {'train_loss_per_batch': [],
                    'train_loss_per_epoch': []}

        if loss_fn is None:
            loss_fn = F.mse_loss

        start_time = time.time()
        for epoch in range(num_epochs):

            model.train()
            for batch_idx, (features, _) in enumerate(train_loader):

                features = features.to(device)

                # FORWARD AND BACK PROP
                logits = model(features)
                loss = loss_fn(logits, features)
                optimizer.zero_grad()

                loss.backward()

                # UPDATE MODEL PARAMETERS
                optimizer.step()

                # LOGGING
                log_dict['train_loss_per_batch'].append(loss.item())

                if not batch_idx % logging_interval:
                    print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f'
                          % (epoch + 1, num_epochs, batch_idx,
                             len(train_loader), loss))

            if not skip_epoch_stats:
                model.eval()

                with torch.set_grad_enabled(False):  # save memory during inference

                    train_loss = compute_epoch_loss_autoencoder(
                        model, train_loader, loss_fn, device)
                    print('***Epoch: %03d/%03d | Loss: %.3f' % (
                        epoch + 1, num_epochs, train_loss))
                    log_dict['train_loss_per_epoch'].append(train_loss.item())

            print('Time elapsed: %.2f min' % ((time.time() - start_time) / 60))

        print('Total Training Time: %.2f min' % ((time.time() - start_time) / 60))
        if save_model is not None:
            torch.save(model.state_dict(), save_model)

        return log_dict


def plot_generated_images(data_loader, model, device,
                          unnormalizer=None,
                          figsize=(20, 2.5), n_images=15, modeltype='autoencoder'):
    fig, axes = plt.subplots(nrows=2, ncols=n_images,
                             sharex=True, sharey=True, figsize=figsize)

    for batch_idx, (features, _) in enumerate(data_loader):

        features = features.to(device)

        color_channels = features.shape[1]
        image_height = features.shape[2]
        image_width = features.shape[3]

        with torch.no_grad():
            if modeltype == 'autoencoder':
                decoded_images = model(features)[:n_images]
            elif modeltype == 'VAE':
                encoder, z_mean, z_log_var, decoded_images = model(features)[:n_images]
            else:
                raise ValueError('`modeltype` not supported')

        orig_images = features[:n_images]
        break

    for i in range(n_images):
        for ax, img in zip(axes, [orig_images, decoded_images]):
            curr_img = img[i].detach().to(torch.device('cpu'))
            if unnormalizer is not None:
                curr_img = unnormalizer(curr_img)

            if color_channels > 1:
                curr_img = np.transpose(curr_img, (1, 2, 0))
                ax[i].imshow(curr_img)
            else:
                ax[i].imshow(curr_img.view((image_height, image_width)), cmap='binary')


def show_image(img):
    img = img.cpu().detach().numpy()
    img = img.transpose(1, 2, 0)  # Convert from CHW to HWC
    plt.imshow(img)
    plt.axis('off')
    plt.show()