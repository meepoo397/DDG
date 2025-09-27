# train.py

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from PIL import Image
from base.torch_utils import distributed as dist
import time
from base.torch_utils import misc
import torch
import base.dnnlib as dnnlib
import string
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from LatentConvNeXtHT import create_latent_convnext_ht_classifier
# 0815第二版
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import tqdm
import torch.nn.functional as F

def create_classifier(num_classes, in_channels, pretrained,**other_kwargs):
    """
    根據參數創建和修改模型
    """
    # 載入模型
    weights = models.EfficientNet_V2_L_Weights.IMAGENET1K_V1
    model = models.efficientnet_v2_l(weights=weights)
    return model
def tensor_preprocess(images_tensor):
    """
    對一個 PyTorch Tensor 執行 EfficientNetV2-L 的預處理。

    Args:
        images_tensor (torch.Tensor): 批次圖片張量，
                                    形狀為 (N, C, H, W)，型態為 uint8，
                                    像素值範圍為 [0, 255]。

    Returns:
        torch.Tensor: 經過預處理的張量，型態為 float32，像素值範圍為 [-1, 1]。
    """
    # 1. 轉換型態與範圍：從 uint8 [0, 255] 轉換為 float32 [0, 1]
    images_float = images_tensor.to(torch.float32) / 255.0

    # 2. 調整大小
    # 使用 bicubic 內插法，並調整到 480x480
    resized_images = F.interpolate(
        images_float,
        size=(480, 480),
        mode='bicubic',
        align_corners=False,
        antialias=True
    )

    # 3. 正規化
    # 獲取 EfficientNetV2-L 預設的正規化參數
    mean = torch.tensor([0.5, 0.5, 0.5], device=images_tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], device=images_tensor.device).view(1, 3, 1, 1)

    # 執行正規化，將 [0, 1] 轉換為 [-1, 1]
    normalized_images = (resized_images - mean) / std

    return normalized_images

def validate(model, encoder, device,batch_size, valid_loader, criterion,batch_num_valid,preprocess_valid):
    valid_loss = 0.0
    correct = 0
    total = 0
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    with torch.no_grad():
        for _ in tqdm.tqdm(range(batch_num_valid//batch_size),total =batch_num_valid//batch_size):
            start_event.record()
            images_meanstd, labels_idx = next(valid_loader)
            # inputs, labels = next(train_dataset_iterator)
            
            # print(f'In training, inputs.shape={inputs.shape}')
            labels_idx = torch.argmax(labels_idx, dim=1)
            # print(f'In training, labels.shape={labels.shape}')
            # inputs = encoder.encode_latents(inputs.to(device))
            # print(f'inputs.shape:{inputs.shape}, labels.shape:{labels.shape}, labels[:10]:{labels[:10]}')
            # print(f'In training, after encoding inputs.shape={inputs.shape}')
            images_meanstd = images_meanstd.to(device)
            labels_idx = labels_idx.to(device)
            end_event.record()
            torch.cuda.synchronize()
            load_time = start_event.elapsed_time(end_event) # milliseconds
            # Step 2: Encoder
            start_event.record()
            images_latent = encoder.encode_latents(images_meanstd)
            end_event.record()
            torch.cuda.synchronize()
            encode_time = start_event.elapsed_time(end_event)

            # Step 3: Decoder
            start_event.record()
            inputs = encoder.decode(images_latent)
            end_event.record()
            torch.cuda.synchronize()
            decode_time = start_event.elapsed_time(end_event)

            # inputs_cpu = inputs.cpu()
    
            # 2. 使用 torchvision 的 transforms 對每張圖片進行處理
            # images_480 = torch.stack([preprocess_valid(transforms.ToPILImage()(img)) for img in inputs_cpu]).to(device)
            start_event.record()
            images_480 = tensor_preprocess(inputs)
            end_event.record()
            torch.cuda.synchronize()
            preprocess_time = start_event.elapsed_time(end_event)
            
            start_event.record()
            outputs = model(images_480)
            end_event.record()
            torch.cuda.synchronize()
            model_time = start_event.elapsed_time(end_event)
            # print(f'outputs:{outputs}')
            
            start_event.record()
            loss = criterion(outputs, labels_idx)
            valid_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels_idx.size(0)
            correct += (predicted == labels_idx).sum().item()
            end_event.record()
            torch.cuda.synchronize()
            metrics_time = start_event.elapsed_time(end_event)
            print(f"Time (ms) - Load: {load_time:.2f} | Encode: {encode_time:.2f} | Decode: {decode_time:.2f} | Preprocess: {preprocess_time:.2f} | Model: {model_time:.2f} | Metrics: {metrics_time:.2f}")
            
    avg_loss = valid_loss / total
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Image classifier training script with command-line arguments")
    
    # 模型和資料相關參數
    parser.add_argument('--num_classes', type=int, default=1000, help='Number of output classes')
    parser.add_argument('--in_channels', type=int, default=3, help='Number of input channels (e.g., 3 for RGB, 4 for RGBA)')
    parser.add_argument('--img_size', type=int, default=224, help='Input image size (e.g., 224 for ResNet)')
    parser.add_argument('--pretrained', action='store_true', help='Use a pre-trained model on ImageNet')
    parser.add_argument('--data_dir', type=str, help='Directory of training data')
    parser.add_argument('--val_data_dir', type=str, help='Directory of validation data')
    
    
    # 訓練相關參數
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='The seed for infinite sampler')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight_decay for optimizer')
    parser.add_argument('--eval_interval', type=int, default=10, help='How often to evaluate and validate')
    parser.add_argument('--save_interval', type=int, default=10, help='How often to save model and optimizer')
    parser.add_argument('--resume_checkpoint', type=str, help='Path of saved model checkpoint')
    parser.add_argument('--save_path', type=str, help='Path to save model checkpoint')
    
    
    args = parser.parse_args()

    weights = models.EfficientNet_V2_L_Weights.IMAGENET1K_V1
    preprocess_valid = weights.transforms()
    # 自動生成訓練集專用的資料增強及預處理轉換
    # 這裡會包含正確的尺寸、隨機裁剪及正規化，並加上資料增強。
    # 具體包含了 RandomResizedCrop、RandomHorizontalFlip、ColorJitter 等。
    # 如果需要更多客製化的增強，可以將其結果 Compose 到自己的 transforms 中。
    preprocess_train = weights.transforms(
        antialias=True,
        interpolation=models.EfficientNet_V2_L_Weights.IMAGENET1K_V1.transforms().interpolation,
    )
    # 注意：`weights.transforms()` 預設已經包含了 ResizedCrop、Flip 和 Normalization。
    # 如果想添加更多 Data Augmentation，應該像這樣：
    preprocess_train_custom = transforms.Compose([
        weights.transforms(),
        transforms.RandomRotation(degrees=15),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.33)),
        # 其他你想添加的增強
    ])
    print("EfficientNetV2-L 的預設預處理參數：")
    print(weights.transforms())

    train_dataset_kwargs      = dict(class_name='training.dataset.ImageFolderDataset',
         path=args.data_dir)
    valid_dataset_kwargs      = dict(class_name='training.dataset.ImageFolderDataset',
         path=args.val_data_dir)
    data_loader_kwargs  = dict(class_name='torch.utils.data.DataLoader', pin_memory=True, num_workers=2, prefetch_factor=2)
    device              = torch.device('cuda')
    
    
    dist.print0("Training with the following parameters:")
    dist.print0(f"Number of classes: {args.num_classes}")
    dist.print0(f"Input channels: {args.in_channels}")
    dist.print0(f"Image size: {args.img_size}x{args.img_size}")
    dist.print0(f"Pre-trained: {args.pretrained}")
    dist.print0(f"Batch size: {args.batch_size}")
    dist.print0(f"Epochs: {args.epochs}")
    dist.print0(f"Learning rate: {args.lr}")
    dist.print0("-" * 30)
    
    # 設置設備 (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dist.print0(f"Using device: {device}")

    if args.in_channels==4:
        encoder_kwargs      = dict(class_name='training.encoders.StabilityVAEEncoder')
    else:
        assert args.in_channels==3
        encoder_kwargs      = dict(class_name='training.encoders.StandardRGBEncoder')

    
    # Load dataset
    dist.print0('Loading dataset...')
    train_dataset_obj = dnnlib.util.construct_class_by_name(**train_dataset_kwargs)
    valid_dataset_obj = dnnlib.util.construct_class_by_name(**valid_dataset_kwargs)
    ref_image, ref_label = train_dataset_obj[0]
    print(f'Checking train_dataset_obj ref_image.shape:{ref_image.shape},ref_image.shape={ref_image.shape}')
    ref_image, ref_label = valid_dataset_obj[0]
    print(f'Checking valid_dataset_obj ref_image.shape:{ref_image.shape},ref_image.shape={ref_label.shape}')
    print(f'len of train dataset:{len(train_dataset_obj)}')
    print(f'len of valid dataset:{len(valid_dataset_obj)}')
    batch_num_train = len(train_dataset_obj)
    batch_num_valid = len(valid_dataset_obj)
    
    state = dnnlib.EasyDict(cur_nimg=0, total_elapsed_time=0)
    seed = args.seed
    
    dist.print0('Creating dataset iterator...')
    train_dataset_sampler = misc.InfiniteSampler(dataset=train_dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed, start_idx=state.cur_nimg)
    train_dataset_iterator = iter(dnnlib.util.construct_class_by_name(dataset=train_dataset_obj, sampler=train_dataset_sampler, batch_size=args.batch_size, **data_loader_kwargs))
    valid_dataset_sampler = misc.InfiniteSampler(dataset=valid_dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed, start_idx=state.cur_nimg)
    valid_dataset_iterator = iter(dnnlib.util.construct_class_by_name(dataset=valid_dataset_obj, sampler=valid_dataset_sampler, batch_size=args.batch_size, **data_loader_kwargs))

    images_from_train_iter,labels_from_train_iter = next(train_dataset_iterator)
    print(f'images_from_train_iter.shape:{images_from_train_iter.shape},labels_from_train_iter.shape:{labels_from_train_iter.shape}')
    images_from_valid_iter,labels_from_valid_iter = next(valid_dataset_iterator)
    print(f'images_from_valid_iter.shape:{images_from_valid_iter.shape},labels_from_valid_iter.shape:{labels_from_valid_iter.shape}')
    
    dist.print0('Setting up encoder...')
    encoder = dnnlib.util.construct_class_by_name(**encoder_kwargs)
    ref_image = encoder.encode_latents(torch.as_tensor(ref_image).to(device).unsqueeze(0))
    
    
    # 建立模型
    model = create_classifier(args.num_classes, args.in_channels, args.pretrained)
    # model = create_latent_convnext_ht_classifier(args.num_classes, args.in_channels, args.pretrained)
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    
    
    # Try to resume from checkpoint
    if args.resume_checkpoint:
        print(f"Loading checkpoint from {args.resume_checkpoint}...")
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        batch_num = checkpoint['batch_num']
        start_epoch = checkpoint['epoch']
        print(f"Resumed from epoch {start_epoch}, batch_num:{batch_num}")
    else:
        start_epoch = 0
        batch_num = 0

    criterion = nn.CrossEntropyLoss()
    dist.print0('Start training...')
    # 訓練迴圈
    running_loss = 0.0
    for epoch in range(start_epoch,args.epochs):
        model.train()
        for batch_idx in range(batch_num_train//args.batch_size):
            # inputs, labels = next(train_dataset_iterator)
            
            # # print(f'In training, inputs.shape={inputs.shape}')
            # labels = torch.argmax(labels, dim=1)
            # # print(f'In training, labels.shape={labels.shape}')
            # labels = labels.to('cuda')
            # inputs = inputs.to('cuda')
            # # inputs = encoder.encode_latents(inputs.to(device))
            # # print(f'inputs.shape:{inputs.shape}, labels.shape:{labels.shape}, labels[:10]:{labels[:10]}')
            # # print(f'In training, after encoding inputs.shape={inputs.shape}')
            
            
            # inputs = encoder.decode(inputs)
            # inputs_cpu = inputs.cpu()
    
            # # 2. 使用 torchvision 的 transforms 對每張圖片進行處理
            # images_224 = torch.stack([preprocess_valid(transforms.ToPILImage()(img)) for img in inputs_cpu])
    
            # # 3. 將處理後的張量移回 CUDA 裝置
            # images_224 = images_224.to(device)
            #########
            images_meanstd, labels_idx = next(train_dataset_iterator)
            images_meanstd = images_meanstd.to(device)
            labels_idx = labels_idx.to(device)
            images_latent = encoder.encode_latents(images_meanstd)
            images_512 = encoder.decode(images_latent.to(device))
            # 2. 使用 torchvision 的 transforms 對每張圖片進行處理
            # images_512 = torch.stack([preprocess(transforms.ToPILImage()(img)) for img in xo_predicted_cpu])
            if labels_idx.dim() > 1:
                labels_idx = torch.argmax(labels_idx, dim=1)
            # images_480 = torch.stack([preprocess_valid(transforms.ToPILImage()(img)) for img in images_512.cpu()])
            # inputs = images_480.to(device)
            images_480 = tensor_preprocess(images_512)
            ##########
            optimizer.zero_grad()
            outputs = model(images_480)
            # print(f'In training, outputs.shape={outputs.shape}')
            # print(f'In training, outputs={outputs}')
            # print(f'In training, labels={labels}')
            loss = criterion(outputs, labels_idx)
            loss.backward()
            # 取得所有參數的總梯度範數
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)  # 計算 L2 範數
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1./2)
            
            # 印出梯度範數
            # print(f'Gradient Norm: {total_norm}')
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            running_loss += loss.item()
            
            # 定期顯示 loss 和驗證結果
            if (batch_num + 1) % args.eval_interval == 0:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                dist.print0(f'-------{now}-------')
                print(f"Epoch [{epoch+1}/{args.epochs}], Batch [{batch_idx+1:5d}/{batch_num_train//args.batch_size:5d}], Loss: {running_loss / args.eval_interval:.4f}")
                # 執行驗證
                valid_loss, valid_acc = validate(model,encoder, device,args.batch_size, valid_dataset_iterator, criterion,batch_num_valid,preprocess_valid)
                print(f"Validation Loss: {valid_loss:.4f}, Accuracy: {valid_acc:.2f}%")
                
                running_loss = 0.0
            if (batch_num + 1) % args.save_interval == 0:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                dist.print0(f'-------{now}-------')
                os.makedirs(args.save_path, exist_ok=True)  # 如果資料夾不存在就建立
                checkpoint_path = os.path.join(args.save_path,f"checkpoint_epoch{epoch}_batch{batch_num+1}.pt")
                torch.save({
                    'epoch': epoch,
                    'batch_num': batch_num + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, checkpoint_path)
                print(f"Model saved to {checkpoint_path}")
            batch_num+=1
    print("Training finished!")

if __name__ == "__main__":
    main()