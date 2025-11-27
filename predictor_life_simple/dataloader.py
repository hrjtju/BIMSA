from itertools import accumulate
import os
from typing import Literal, Tuple, List
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
        self.arr_list = [np.load(i) for i in file_list]
        self.data_len_ls = [i.shape[0] for i in self.arr_list]
        self.len_prefix_sum = list(accumulate(self.data_len_ls))
        
    def locate_idx(self, i: int) -> int:
        res_arr = [i-k for k in self.len_prefix_sum if k <= i]
        return len(res_arr) - 1
    
    def __len__(self) -> int:
        return self.len_prefix_sum[-1]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_idx = self.locate_idx(idx)
        file_path = self.file_list[file_idx]
        data = np.load(file_path)  # Shape: [T, N, N]
        t = np.random.randint(0, data.shape[0] - 1)
        x, x_1 = map(lambda x: torch.tensor(x, dtype=torch.float32), 
                          [data[t], data[t + 1]])
        x = (x / (dim:=(255 if x.max() > 100 else 1)))[None, ...]
        y = (x_1 / dim)
        return x, y

def get_dataloader(data_dir, 
                   batch_size: int, 
                   shuffle: bool = True, 
                   num_workers: int = 0, 
                   split: Literal["train", "test", "all"] = "train"
                   ):
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
    test_files = all_files[::5]
    if split == "train":
        file_list = [i for i in all_files if i not in test_files]
    elif split == "test":
        file_list = test_files
    elif split == "all":
        file_list = all_files
    else:
        raise NotImplementedError(f"Argument 'split'(current value {split}) only accepts 'train', 'test' or 'all'.")
    
    dataset = LifeGameDataset(file_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


if __name__ == "__main__":
    data_dir = r"D:\Internship\bimsa\predictor_life_simple\datasets\200-100-B3_S23"
    dataset = LifeGameDataset(os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy'))
    
    print(dataset.__len__)
    print(dataset.data_len_ls)
    print(dataset.len_prefix_sum)