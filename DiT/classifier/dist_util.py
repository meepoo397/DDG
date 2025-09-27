"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 2

SETUP_RETRY_COUNT = 3


def setup_dist():
    if dist.is_initialized():
        return

    local_rank = int(os.environ["LOCAL_RANK"])
    th.cuda.set_device(local_rank)

    # Init process group
    backend = "nccl" if th.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    logger.log(f"In load_state_dict, rank = {dist.get_rank()}...")
    DUBUG=False
    import torch
    def check_dist_state():
        print(f"[Rank {torch.distributed.get_rank()}] is_initialized: {torch.distributed.is_initialized()}")
        print(f"[Rank {torch.distributed.get_rank()}] backend: {torch.distributed.get_backend()}")
        print(f"[Rank {torch.distributed.get_rank()}] world_size: {torch.distributed.get_world_size()}")
        print(f"[Rank {torch.distributed.get_rank()}] rank: {torch.distributed.get_rank()}")
        print(f"[Rank {torch.distributed.get_rank()}] cuda available: {torch.cuda.is_available()}")
        print(f"[Rank {torch.distributed.get_rank()}] current device: {torch.cuda.current_device()}")
        print(f"[Rank {torch.distributed.get_rank()}] device count: {torch.cuda.device_count()}")
        print(f"[Rank {torch.distributed.get_rank()}] default_pg: {torch.distributed.distributed_c10d._get_default_group()}")

    # check_dist_state()
    chunk_size = 2 ** 30  # MPI has a relatively small size limit
    if dist.get_rank() == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
        data_tensor = th.ByteTensor(list(data))
        num_chunks = len(data_tensor) // chunk_size
        if len(data_tensor) % chunk_size:
            num_chunks += 1
    else:
        data_tensor = None
        num_chunks = 0
    
    if dist.get_rank() == 0:
        temp = th.tensor([num_chunks], dtype=th.long).cuda()
        # print(f'[Rank {dist.get_rank()}] Distributing temp')
    else:
        temp = th.zeros(1, dtype=th.long).cuda()
        # print(f'[Rank {dist.get_rank()}] Recieving temp')
    dist.broadcast(temp, 0)
    num_chunks = int(temp.item())

    chunks = []
    for i in range(0, num_chunks):
        if dist.get_rank() == 0:
            # print(f'[Rank {dist.get_rank()}] Distributing chunk {i}')
            chunk = data_tensor[i * chunk_size : (i + 1) * chunk_size].cuda()
            shape = th.tensor([len(chunk)], dtype=th.long).cuda()
        else:
            # print(f'[Rank {dist.get_rank()}] Recievinging chunk {i}')
            shape = th.zeros(1, dtype=th.long).cuda()
        dist.broadcast(shape, 0)
        chunk_len = shape.item()

        if dist.get_rank() != 0:
            chunk = th.empty(int(chunk_len), dtype=th.uint8).cuda()
        
        dist.broadcast(chunk, 0)
        chunks.append(chunk)
    
    full = th.cat(chunks, dim=0)
    byte_data = bytes(full.tolist())
    return th.load(io.BytesIO(byte_data), **kwargs)
    
    # if MPI.COMM_WORLD.Get_rank() == 0:
    #     with bf.BlobFile(path, "rb") as f:
    #         data = f.read()
    #     num_chunks = len(data) // chunk_size
    #     if len(data) % chunk_size:
    #         num_chunks += 1
    #     MPI.COMM_WORLD.bcast(num_chunks)
    #     for i in range(0, len(data), chunk_size):
    #         MPI.COMM_WORLD.bcast(data[i : i + chunk_size])
    # else:
    #     num_chunks = MPI.COMM_WORLD.bcast(None)
    #     data = bytes()
    #     for _ in range(num_chunks):
    #         data += MPI.COMM_WORLD.bcast(None)

    # return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
