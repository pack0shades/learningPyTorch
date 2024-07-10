import torch
import torchvision.transforms as transforms
from PIL import Image
import os  # for loading file paths
import pandas as pd
import spacy  # for tokenisation
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

nlp = spacy.load("en_core_web_sm")


def save_checkpoint(state, filename='check.pth.tar'):
    print('saving checkpoint...\n')
    torch.save(state, filename)
    print('CHECKPOINT SAVED!!!!')


def load_checkpoint(checkpoint, model, optimizer):
    print('=> loading checkpoint...\n')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    step = checkpoint["step"]
    print('LOADED!!!')
    return step


def print_examples(model, device, dataset, image_folder="examples"):
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    model.eval()

    # Loop through each image in the folder
    for image_name in os.listdir(image_folder):
        # Create the full path to the image
        image_path = os.path.join(image_folder, image_name)

        # Open and transform the image
        image = transform(Image.open(image_path).convert("RGB")).unsqueeze(0)

        # Generate caption for the image
        caption = " ".join(model.caption_image(image.to(device), dataset.vocab))

        # Print the image name and its caption
        print(f"Image_name: {image_name}")
        print(f"OUTPUT caption: {caption}")
        print('------------------------------------------------------------')
    model.train()


class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_thres = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in nlp.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4  # 0, 1, 2, 3 are already used for pad sos eos and unk tokens

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1

                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_thres:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi['<UNK>']
            for token in tokenized_text
        ]


class FlickDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform
        self.imgs = self.df['image']
        self.captions = self.df['caption']

        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.imgs[index]
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        numericalized_caption = [self.vocab.stoi['<SOS>']]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi['<EOS>'])

        return img, torch.tensor(numericalized_caption)


class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return imgs, targets

def loader(
        root_folder,
        annotation_file,
        transform,
        batch_size=32,
        num_workers=8,
        shuffle=True,
        pin_memory=True
):
    dataset = FlickDataset(root_folder, annotation_file, transform=transform)

    pad_idx = dataset.vocab.stoi["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx)
    )

    return loader, dataset


if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    data_loader, dataset = loader(
        "image_captions/images/", "image_captions/captions.txt", transform=transform
    )

    for idx, (imgs, captions) in enumerate(data_loader):
        print(imgs.shape)
        print(captions.shape)