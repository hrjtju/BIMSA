import os
from typing import Tuple, List
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class LifeGameDataset(Dataset):
    def __init__(self, file_list: List[str]):
        """
        Args:
            file_list (List[str]): List of file paths to use in the dataset.
        """
        self.file_list = file_list

    def __len__(self) -> int:
        return len(self.file_list) * 1000

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_idx = idx // 1000
        file_path = self.file_list[file_idx]
        data = np.load(file_path)  # Shape: [T, N, N]
        t = np.random.randint(0, data.shape[0] - 1)
        x = data[t]
        y = data[t + 1]
        x = (torch.tensor(x, dtype=torch.float32)[None, ...] / 255)
        y = (torch.tensor(y, dtype=torch.float32)[None, ...] / 255)
        return x, y

def get_dataloader(data_dir, batch_size: int, shuffle: bool = True, num_workers: int = 0, split: str = 'train', split_ratio: float = 0.8):
    """
    Creates a DataLoader for the LifeGameDataset with train/test split.

    Args:
        data_dir (str): Path to the directory containing the dataset files.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of subprocesses to use for data loading.
        split (str): 'train' or 'test'.
        split_ratio (float): Ratio of training data.

    Returns:
        DataLoader: DataLoader for the dataset.
    """
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')]
    all_files.sort()  # Ensure deterministic split
    split_idx = int(len(all_files) * split_ratio)
    if split == 'train':
        file_list = all_files[:split_idx]
    else:
        file_list = all_files[split_idx:]
    dataset = LifeGameDataset(file_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader