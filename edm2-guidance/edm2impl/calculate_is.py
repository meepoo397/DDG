## Use implementation from https://github.com/sbarratt/inception-score-pytorch/
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
import tqdm

from torchvision.models.inception import inception_v3
from torchvision.datasets import ImageFolder
from torchvision import transforms

import numpy as np
from scipy.stats import entropy
import argparse
import os

def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in tqdm.tqdm(enumerate(dataloader, 0), total=len(dataloader)):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


class IgnoreLabelDataset(torch.utils.data.Dataset):
    def __init__(self, folder_dataset, max_images=None):
        self.folder_dataset = folder_dataset
        self.max_images = max_images

    def __getitem__(self, index):
        return self.folder_dataset[index][0]

    def __len__(self):
        if self.max_images:
            return min(self.max_images, len(self.folder_dataset))
        return len(self.folder_dataset)
#python ./edm2-guidance/edm2impl/calculate_is.py --data_root D:\temp\edm2\imagenet-val --cuda --batch_size 64 --splits 10 --resize --max_images 50000
def main():
    parser = argparse.ArgumentParser(description='Calculate Inception Score')
    parser.add_argument('--data_root', type=str, default='imagenet512/', help='Path to image dataset root')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA (GPU) if available')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for Inception')
    parser.add_argument('--splits', type=int, default=10, help='Number of splits for IS computation')
    parser.add_argument('--resize', action='store_true', help='Resize images to 299x299 before Inception')
    parser.add_argument('--max_images', type=int, default=None, help='Max number of images to use for IS')
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    img_dataset = ImageFolder(args.data_root, transform=transform)
    ignore_label_dataset = IgnoreLabelDataset(img_dataset, max_images=args.max_images)

    print(f"Calculating Inception Score using {len(ignore_label_dataset)} images...")
    mean, std = inception_score(
        ignore_label_dataset,
        cuda=args.cuda,
        batch_size=args.batch_size,
        resize=args.resize,
        splits=args.splits
    )
    print(f"Inception Score: {mean:.4f} ± {std:.4f}")

if __name__ == "__main__":
    main()