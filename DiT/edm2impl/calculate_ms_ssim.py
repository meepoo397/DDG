import torch
import random
from PIL import Image
import torchvision.transforms as T
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
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
# def dataset_ms_ssim_example(dataset, num_pairs=1000, image_size=(256, 256), device="cuda", seed=42, batch_size=32):
#     """
#     Compute average MS-SSIM within a dataset (internal diversity metric).

#     Args:
#         dataset: list of file paths or list of torch tensors (C,H,W in [0,1])
#         num_pairs: number of random pairs to compare
#         image_size: resize all images to this size
#         device: "cuda" or "cpu"
#         seed: random seed for reproducibility
#         batch_size: number of pairs to compute per batch

#     Returns:
#         float: average MS-SSIM score (lower = higher diversity)
#     """
#     random.seed(seed)
#     torch.manual_seed(seed)

#     transform = T.Compose([
#         T.Resize(image_size),
#         T.ToTensor()
#     ])

#     # Preload and preprocess all images once
#     processed_images = []
#     for img in dataset:
#         if isinstance(img, str):
#             img = transform(Image.open(img).convert("RGB"))
#         processed_images.append(img.unsqueeze(0))  # Add batch dim

#     processed_images = torch.cat(processed_images).to(device)

#     ms_ssim_metric = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)

#     scores = []
#     n = len(processed_images)

#     # Process in batches
#     for start in range(0, num_pairs, batch_size):
#         pairs = [random.sample(range(n), 2) for _ in range(min(batch_size, num_pairs - start))]
#         imgs1 = torch.stack([processed_images[i1] for i1, _ in pairs])
#         imgs2 = torch.stack([processed_images[i2] for _, i2 in pairs])

#         batch_scores = ms_ssim_metric(imgs1, imgs2)  # Vectorized
#         scores.append(batch_scores.mean().item())

#     return sum(scores) / len(scores)

# Example usage:
# dataset = ["gen/img1.png", "gen/img2.png", ...]
# diversity_score = dataset_ms_ssim(dataset, num_pairs=1000)
# print("MS-SSIM (lower is better diversity):", diversity_score)


def dataset_ms_ssim(imgs, num_pairs=1000, image_size=(256, 256), device="cuda", seed=42, batch_size=32):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    random.seed(seed)
    torch.manual_seed(seed)
    
    # Set up dataloader with random data(seed specified)
    g = torch.Generator()
    g.manual_seed(seed)
    dataloader = torch.utils.data.DataLoader(
        imgs,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
    )
    dataloader_iter = iter(dataloader)
    ms_ssim_metric = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    if len(range(0, num_pairs, batch_size)) >len(dataloader):
        print(f'Num of pairs:{num_pairs} run over entire dataloader with len {len(dataloader)}, reduce to use len {len(dataloader)} pairs only.')
        num_pairs = len(dataloader)*batch_size
    scores = []
    for start in tqdm.tqdm(range(0, num_pairs, batch_size)):
        batch = next(dataloader_iter).to(device)
        n=min(batch.shape[0], num_pairs - start)
        batch=batch[:n]
        
        pairs = [random.sample(range(n), 2) for _ in range(n)]
        imgs1 = torch.stack([batch[i1] for i1, _ in pairs]).to(device)
        imgs2 = torch.stack([batch[i2] for _, i2 in pairs]).to(device)
        # print(f"MS-SSIM metric device: {ms_ssim_metric.parameters().device}")
        # print(f"Batch device: {batch.device}")
        # print(f"imgs1 device: {imgs1.device}")
        # print(f"imgs2 device: {imgs2.device}")
        batch_scores = ms_ssim_metric(imgs1, imgs2)  # Vectorized
        scores.append(batch_scores.mean().item())

    return sum(scores) / len(scores)


class IgnoreLabelDataset(torch.utils.data.Dataset):
    def __init__(self, folder_dataset, max_images=None, transform=None):
        self.folder_dataset = folder_dataset
        self.max_images = max_images
        self.transform = transform

    def __getitem__(self, index):
        img = self.folder_dataset[index][0]  # 只取圖
        if self.transform:
            img = self.transform(img)       # 套用 transform
        return img

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
    parser.add_argument('--num_pairs', type=int, default=1000, help='Number of pairs sampled for averaging ssim')
    parser.add_argument('--size', type=int,default=256, help='Resize images to a specific size before compute ssim')
    parser.add_argument('--seed', type=int,default=42, help='The seed for random sampling pairs')
    parser.add_argument('--max_images', type=int, default=None, help='Max number of images to use for IS')
    args = parser.parse_args()

    transform = T.Compose([
        T.Resize(args.size),
        T.CenterCrop(args.size),
        T.ToTensor()
    ])

    img_dataset = ImageFolder(args.data_root)  # 不要 transform 交給它
    ignore_label_dataset = IgnoreLabelDataset(
        img_dataset,
        max_images=args.max_images,
        transform=transform  # 在這裡統一套 transform
    )
    print(f"Calculating Multi Scale SSIM using {len(ignore_label_dataset)} images...")
    
    ms_ssim = dataset_ms_ssim(
        ignore_label_dataset,
        num_pairs=args.num_pairs,
        image_size=(args.size, args.size),
        device="cuda" if args.cuda else "cpu",
        seed=42,
        batch_size=args.batch_size,
    )
    print(f"MS SSIM: {ms_ssim:.6f}")

if __name__ == "__main__":
    main()