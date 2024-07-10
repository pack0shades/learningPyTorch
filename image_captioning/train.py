import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from loader import loader, save_checkpoint, load_checkpoint, print_examples
from model import CNNtoRNN
import os
from tqdm import tqdm
imagefile = os.path.abspath('image_captions/images/')
captionsfile = os.path.abspath('image_captions/captions.txt')
def train():
    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_loader, dataset = loader(
        root_folder=imagefile,
        annotation_file=captionsfile,
        transform=transform,
        num_workers=2
    )

    torch.backends.cudnn.benchmark = True

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    load_model = False
    save_model = False
    train_CNN = False

    # Hyperparameter
    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 500

    # tensorboard
    writer = SummaryWriter("runs/flickr")
    step = 0

    # initializing model
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for name, param in model.CNN.inception.named_parameters():
        if "fc.weight" in name or "fc.bias" in name:
            param.requires_grad = True
        else:
            param.requires_grad = train_CNN


    if load_model:
        step = load_checkpoint(torch.load('my_checkpoint.pth.tar'), model, optimizer)

    model.train()

    for epoch in range(num_epochs):
        print_examples(model, DEVICE, dataset)

        if save_model:
            checkpoint = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step
            }
            save_checkpoint(checkpoint)

        for idx, (imgs, captions) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):
            imgs = imgs.to(DEVICE)
            captions = captions.to(DEVICE)

            outputs = model(imgs, captions[:-1])
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))

            writer.add_scalar('Training loss', loss.item(), global_step=step)
            step += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == '__main__':
    train()
