from email.policy import default
from base.calculate_metrics import calculate_stats_for_iterable
import tqdm
from base.torch_utils import distributed as dist
import os
import click
import torch
import numpy as np
from base.dataset_tool import make_transform
from torchvision import transforms

import os
import PIL.Image
import numpy as np
from torch.utils.data import Dataset

import os
import PIL.Image
import numpy as np
from torch.utils.data import Dataset
from base.dataset_tool import make_transform  # 你原本的 transform 工具
import scipy.linalg
import pandas as pd
import psutil

class ImageFolderRawDataset(Dataset):
    def __init__(self, path: str, transform_name=None, output_width=None, output_height=None):
        self.path = path
        self.transform_name = transform_name
        self.output_width = output_width
        self.output_height = output_height
        self._transform = None  # 先不建立 transform，延後建立

        self._valid_exts = ['.jpg', '.jpeg', '.png', '.bmp']
        self._all_filenames = []
        for root, _, files in os.walk(self.path):
            for fname in files:
                if os.path.splitext(fname)[1].lower() in self._valid_exts:
                    self._all_filenames.append(os.path.join(root, fname))
        if len(self._all_filenames) == 0:
            raise ValueError(f'Found no valid images in {self.path}')
        self._all_filenames.sort()

    def __len__(self):
        return len(self._all_filenames)

    def __getitem__(self, idx):
        if self._transform is None and self.transform_name is not None:
            base_transform = make_transform(self.transform_name, self.output_width, self.output_height)
            # 包裝在後面加上必要的轉換
            self._transform = transforms.Compose([
                base_transform,
                transforms.ToTensor(),  # ✅ numpy to torch.FloatTensor in [0, 1]
                transforms.Normalize([0.5]*3, [0.5]*3),  # ✅ normalize to [-1, 1]
            ])
        fname = self._all_filenames[idx]
        try:
            with open(fname, 'rb') as f:
                img = PIL.Image.open(f).convert('RGB')
            img = np.array(img)
        except Exception as e:
            dist.print0(f"Warning: Failed to load image {fname}: {e}")
            # 用一張黑色空白圖替代，避免中斷
            img = np.zeros((self.output_height, self.output_width, 3), dtype=np.uint8)
        fname = self._all_filenames[idx]
        try:
            with open(fname, 'rb') as f:
                img = PIL.Image.open(f).convert('RGB')
            img = np.array(img)
        except Exception as e:
            dist.print0(f"[ERROR] Failed to load image: {fname} => {e}")
            img = np.zeros((self.output_height, self.output_width, 3), dtype=np.uint8)

        if self._transform is not None:
            try:
                img = self._transform(img)
            except Exception as e:
                dist.print0(f"Warning: transform failed for {fname}: {e}")
                img = torch.zeros(3, self.output_height, self.output_width)  # torch tensor 的黑圖

        return img



def calculate_stats_for_files_with_transform(
    image_path,             # Path to a directory or ZIP file containing the images.
    num_images      = None, # Number of images to use. None = all available images.
    seed            = 0,    # Random seed for selecting the images.
    max_batch_size  = 64,   # Maximum batch size.
    num_workers     = 0,    # How many subprocesses to use for data loading.
    prefetch_factor = 2,    # Number of images loaded in advance by each worker.
    verbose         = True, # Enable status prints?
    image_size=299,         # Resize all image into (image_size,image_size)
    **stats_kwargs,         # Arguments for calculate_stats_for_iterable().
):
    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # List images.
    if verbose:
        dist.print0(f'Loading images from {image_path} ...')
    dataset_obj = ImageFolderRawDataset(
        path=image_path,
        transform_name='center-crop-dhariwal',
        output_width=image_size,
        output_height=image_size
    )
    if num_images is not None and len(dataset_obj) < num_images:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but expected at least {num_images}')
    if len(dataset_obj) < 2:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but need at least 2 to compute statistics')

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Divide images into batches.
    num_batches = max((len(dataset_obj) - 1) // (max_batch_size * dist.get_world_size()) + 1, 1) * dist.get_world_size()
    rank_batches = np.array_split(np.arange(len(dataset_obj)), num_batches)[dist.get_rank() :: dist.get_world_size()]
    data_loader = torch.utils.data.DataLoader(dataset_obj, batch_sampler=rank_batches,
        num_workers=num_workers, prefetch_factor=(prefetch_factor if num_workers > 0 else None))

    # Return an interable for calculating the statistics.
    return calculate_stats_for_iterable(image_iter=data_loader, verbose=verbose, **stats_kwargs)
def compute_fid(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean, _ = scipy.linalg.sqrtm(sigma1 @ sigma2, disp=False)

    # 修正數值誤差
    if not np.isfinite(covmean).all():
        covmean = np.eye(sigma1.shape[0])
    
    fid = np.sum(diff**2) + np.trace(sigma1 + sigma2 - 2 * np.real(covmean))
    return float(fid)
def compute_fid_gpu_test(mu1, sigma1, mu2, sigma2, device='cuda', eps=1e-6, num_iter=10):
    # ----- CPU版（scipy） -----
    diff_cpu = mu1 - mu2
    mean_diff_cpu = np.sum(diff_cpu**2)

    covmean_cpu, _ = scipy.linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean_cpu).all():
        covmean_cpu = np.eye(sigma1.shape[0])

    trace_cpu = np.trace(sigma1 + sigma2 - 2 * np.real(covmean_cpu))
    fid_cpu = mean_diff_cpu + trace_cpu

    # ----- 轉為torch tensor -----
    if isinstance(mu1, np.ndarray): mu1 = torch.from_numpy(mu1).float()
    if isinstance(sigma1, np.ndarray): sigma1 = torch.from_numpy(sigma1).float()
    if isinstance(mu2, np.ndarray): mu2 = torch.from_numpy(mu2).float()
    if isinstance(sigma2, np.ndarray): sigma2 = torch.from_numpy(sigma2).float()

    mu1 = mu1.to(device)
    sigma1 = sigma1.to(device)
    mu2 = mu2.to(device)
    sigma2 = sigma2.to(device)

    # 加入正則項確保正定性
    sigma1_reg = sigma1 + eps * torch.eye(sigma1.shape[0], device=device, dtype=sigma1.dtype)
    sigma2_reg = sigma2 + eps * torch.eye(sigma2.shape[0], device=device, dtype=sigma2.dtype)

    # 均值差平方和
    diff = mu1 - mu2
    mean_diff_gpu = torch.sum(diff ** 2)

    # sqrtm 部分（使用 Newton-Schulz）
    try:
        # sqrt_sigma1 = matrix_sqrt_newton_schulz(sigma1_reg, num_iter=num_iter)
        sqrt_sigma1 = matrix_sqrt_eigh(sigma1_reg)
        intermediate = sqrt_sigma1 @ sigma2_reg @ sqrt_sigma1
        if not torch.isfinite(intermediate).all():
            print("intermediate matrix has NaN or Inf values!")

        # covmean_gpu = matrix_sqrt_newton_schulz(intermediate, num_iter=num_iter)
        covmean_gpu = matrix_sqrt_eigh(intermediate)
    except Exception:
        print("警告: 使用近似")
        covmean_gpu = torch.sqrt(torch.trace(sigma1_reg) * torch.trace(sigma2_reg)) / sigma1.shape[0] * torch.eye(sigma1.shape[0], device=device, dtype=sigma1.dtype)

    # 確保 covmean 為實數
    if torch.is_complex(covmean_gpu):
        covmean_gpu = covmean_gpu.real

    # trace 計算
    trace_gpu = torch.trace(sigma1_reg) + torch.trace(sigma2_reg) - 2 * torch.trace(covmean_gpu)

    # ----- 印出數值比較 -----
    print("== FID 分項差異比較 ==")
    print(f"mean_diff (CPU): {mean_diff_cpu:.6f}")
    print(f"mean_diff (GPU): {mean_diff_gpu.item():.6f}")
    print(f"→ abs diff: {abs(mean_diff_cpu - mean_diff_gpu.item()):.6f}\n")

    print(f"covmean trace (CPU): {float(np.trace(covmean_cpu)):.6f}")
    print(f"covmean trace (GPU): {torch.trace(covmean_gpu).item():.6f}")
    print(f"→ abs diff: {abs(float(np.trace(covmean_cpu)) - torch.trace(covmean_gpu).item()):.6f}\n")

    print(f"trace_sum (CPU): {trace_cpu:.6f}")
    print(f"trace_sum (GPU): {trace_gpu.item():.6f}")
    print(f"→ abs diff: {abs(trace_cpu - trace_gpu.item()):.6f}\n")

    fid_gpu = mean_diff_gpu + trace_gpu
    fid_gpu = torch.clamp(fid_gpu, min=0.0)

    print(f"FID (CPU total): {fid_cpu:.6f}")
    print(f"FID (GPU total): {fid_gpu.item():.6f}")
    print(f"→ abs diff: {abs(fid_cpu - fid_gpu.item()):.6f}")

    return fid_gpu.item()
def matrix_sqrt_eigh(A):
    # A 為對稱正定矩陣 (B, N, N)
    eigvals, eigvecs = torch.linalg.eigh(A)  # GPU-safe
    # 避免 sqrt 負數
    sqrt_eigvals = torch.sqrt(torch.clamp(eigvals, min=1e-10))
    return eigvecs @ torch.diag_embed(sqrt_eigvals) @ eigvecs.transpose(-2, -1)
def compute_fid_gpu(mu1, sigma1, mu2, sigma2, device='cuda', eps=1e-6, num_iter=10):
    # ----- 轉為torch tensor -----
    if isinstance(mu1, np.ndarray): mu1 = torch.from_numpy(mu1).float()
    if isinstance(sigma1, np.ndarray): sigma1 = torch.from_numpy(sigma1).float()
    if isinstance(mu2, np.ndarray): mu2 = torch.from_numpy(mu2).float()
    if isinstance(sigma2, np.ndarray): sigma2 = torch.from_numpy(sigma2).float()

    mu1 = mu1.to(device)
    sigma1 = sigma1.to(device)
    mu2 = mu2.to(device)
    sigma2 = sigma2.to(device)

    # 加入正則項確保正定性
    sigma1_reg = sigma1 + eps * torch.eye(sigma1.shape[0], device=device, dtype=sigma1.dtype)
    sigma2_reg = sigma2 + eps * torch.eye(sigma2.shape[0], device=device, dtype=sigma2.dtype)

    # 均值差平方和
    diff = mu1 - mu2
    mean_diff_gpu = torch.sum(diff ** 2)

    # sqrtm 部分（使用 Newton-Schulz）
    try:
        sqrt_sigma1 = matrix_sqrt_eigh(sigma1_reg)
        intermediate = sqrt_sigma1 @ sigma2_reg @ sqrt_sigma1
        if not torch.isfinite(intermediate).all():
            print("intermediate matrix has NaN or Inf values!")
        covmean_gpu = matrix_sqrt_eigh(intermediate)
    except Exception:
        print("警告: 使用近似")
        covmean_gpu = torch.sqrt(torch.trace(sigma1_reg) * torch.trace(sigma2_reg)) / sigma1.shape[0] * torch.eye(sigma1.shape[0], device=device, dtype=sigma1.dtype)

    # 確保 covmean 為實數
    if torch.is_complex(covmean_gpu):
        covmean_gpu = covmean_gpu.real

    # trace 計算
    trace_gpu = torch.trace(sigma1_reg) + torch.trace(sigma2_reg) - 2 * torch.trace(covmean_gpu)

    fid_gpu = mean_diff_gpu + trace_gpu
    fid_gpu = torch.clamp(fid_gpu, min=0.0)
    return fid_gpu.item()
def build_matrix(
    root_directory_path,       #Path of all class directory
    class_stats_path,
    output_path,
    batch_size=64,
):
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    if not os.path.isdir(root_directory_path):
        raise ValueError(f'root_directory_path:{root_directory_path} is not a direcory.')
    classes = [dirname for dirname in os.listdir(root_directory_path) if os.path.isdir(os.path.join(root_directory_path,dirname))]
    # classes = classes[:3]
    # classes = classes[620:]  # or 崩潰當下那個 class

    dist.print0(f'Found classes:{classes[:5]}...')
    if len(classes)==0:
        raise ValueError(f'Found no class directories in {root_directory_path}')
    
    os.makedirs(class_stats_path, exist_ok=True)
    all_stats = {}
    for class_name in tqdm.tqdm(classes, total=len(classes)):
        # stats_path = os.path.join(class_stats_path, f"{class_name}.npz")
        # if os.path.exists(os.path.join(class_stats_path, f"{class_name}_mu.npy")) and os.path.exists(os.path.join(class_stats_path, f"{class_name}_sigma.npy")):
        #     continue  # 已經處理過的就跳過

        stats_iter = calculate_stats_for_files_with_transform(
            os.path.join(root_directory_path, class_name),
            verbose=False,
            metrics=['fid'],
            max_batch_size=batch_size
        )
        r = None
        for item in stats_iter:
            r = item

        if dist.get_rank() == 0:
            if r is None:
                raise RuntimeError(f"No stats produced for class {class_name}")
            mu = np.array(r.stats['fid']['mu'])
            sigma = np.array(r.stats['fid']['sigma'])
            # np.savez(stats_path, mu=mu, sigma=sigma)
            # np.save(os.path.join(class_stats_path, f"{class_name}_mu.npy"), mu)
            # np.save(os.path.join(class_stats_path, f"{class_name}_sigma.npy"), sigma)
            all_stats[class_name] = {
                # 'mu': data['mu'],
                # 'sigma': data['sigma']
                'mu': mu,
                'sigma': sigma
            }
            mem = psutil.virtual_memory()
            print_gpu_memory(class_name=class_name)
            print(f"[{class_name}] Used: {mem.used / 1024**3:.2f} GB, Free: {mem.available / 1024**3:.2f} GB")


        torch.cuda.empty_cache()
        torch.distributed.barrier()

    # -------- 建立 FID 矩陣 --------
    if dist.get_rank() == 0:
        dist.print0('Prepare to build the matrix')
        class_names = classes
        n = len(class_names)
        fid_matrix = np.zeros((n, n))
        dist.print0('Start building the matrix')
        # all_stats = {}
        # for class_name in tqdm.tqdm(classes, desc='Loading stats'):
        #     stats_path = os.path.join(class_stats_path, f"{class_name}.npz")
        #     data = np.load(stats_path)  # 或可加 mmap_mode='r'
        #     all_stats[class_name] = {
        #         'mu': data['mu'],
        #         'sigma': data['sigma']
        #     }
        #     mem = psutil.virtual_memory()
        #     print(f"[{class_name}] Used: {mem.used / 1024**3:.2f} GB, Free: {mem.available / 1024**3:.2f} GB")

        for i in tqdm.tqdm(range(n),total=n):
            # dist.print0(f'Loading {i},class_name={class_names[i]}')
            # mu1, sigma1 = load_class_stats(class_names[i],class_stats_path)
            mu1 = all_stats[class_names[i]]['mu']
            sigma1 = all_stats[class_names[i]]['sigma']
            for j in range(n):
                if i == j:
                    fid_matrix[i, j] = 0.0
                else:
                    # mu2, sigma2 = load_class_stats(class_names[j],class_stats_path)
                    mu2 = all_stats[class_names[j]]['mu']
                    sigma2 = all_stats[class_names[j]]['sigma']
                    # dist.print0('Start computing fid')
                    # fid = compute_fid(mu1, sigma1, mu2, sigma2)
                    fid = compute_fid_gpu(mu1, sigma1, mu2, sigma2)
                    # dist.print0('Finish computing fid')
                    fid_matrix[i, j] = fid

        df = pd.DataFrame(fid_matrix, index=class_names, columns=class_names)
        print(df)
        df.to_csv(output_path)
        dist.print0(f'Output the result matrix at {output_path}')
    
    torch.distributed.destroy_process_group()
def print_gpu_memory(class_name=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[{class_name}] GPU Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    else:
        print(f"[{class_name}] CUDA not available")
def load_class_stats(class_name,class_stats_path):
    # stats_path = os.path.join(class_stats_path, f"{class_name}.npz")
    mu_path = os.path.join(class_stats_path, f"{class_name}_mu.npy")
    sigma_path = os.path.join(class_stats_path, f"{class_name}_mu.npy")
    dist.print0(f'Try to load {mu_path},{sigma_path}')
    if not os.path.exists(mu_path):
        raise FileNotFoundError(f"Stats not found for class {class_name},at {mu_path}")
    if not os.path.exists(sigma_path):
        raise FileNotFoundError(f"Stats not found for class {class_name},at {sigma_path}")
    try:
        # data = np.load(stats_path)
        # 讀取階段
        mu = np.load(os.path.join(class_stats_path, f"{class_name}_mu.npy"))
        sigma = np.load(os.path.join(class_stats_path, f"{class_name}_sigma.npy"))

    except e:
        raise ValueError(f'Failed to load {stats_path}, {e}')

    # return data['mu'], data['sigma']
    return mu,sigma
@click.command()
@click.option('--path', 'path',     help='Path to the root image directory', metavar='PATH',type=str, required=True)
@click.option('--class_stats_path', 'class_stats_path',     help='Path to the cache mean,covariance matrix of each class', metavar='PATH',type=str, required=True)
@click.option('--output_path', 'output_path',     help='Path to the output fid matrix', metavar='PATH',type=str,default='./fid_matrix.csv', show_default=True)
@click.option('--batch', 'max_batch_size',  help='Maximum batch size', metavar='INT',                       type=click.IntRange(min=1), default=64, show_default=True)
def cmdline(path,class_stats_path,output_path,max_batch_size):
    build_matrix(path,class_stats_path,output_path,batch_size=max_batch_size)

if __name__ == "__main__":
    cmdline()