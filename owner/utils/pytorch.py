"""Utilities for Pytorch
"""
import multiprocessing
import torch

IGNORE_VALUE = -100


def get_num_workers() -> int:
    """Returns the number of workers that can be used. A good approximation 
        is the number of cores of the CPU.
    Returns:
        int: number of workers.
    """
    return multiprocessing.cpu_count()


def remap_values(x: torch.Tensor, remapping: torch.Tensor) -> torch.Tensor:
    """Remap values from x
    Args:
        x (torch.Tensor): values to remap
        remapping (torch.Tensor): translation table. Shape [2(x_val, remap_val), length]
    Returns:
        torch.Tensor: remapped values
    """
    index = torch.bucketize(x.ravel(), remapping[0])
    return remapping[1][index].reshape(x.shape)
