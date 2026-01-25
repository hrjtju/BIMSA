from itertools import accumulate
import os
from typing import Literal, Tuple, List
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as v2

transform = v2.Compose([
    v2.RandomErasing(0.8, (5e-4, 1e-3), value=0.5),
    v2.RandomErasing(0.8, (5e-4, 1e-3), value=0.5),
])

def precompute_transform_table(K, p, M=2**32):
    """
    compute the thresholds for transforming uniform integers to time steps
    """
    total_weight = 1 - (1-p)**(K+1)
    thresholds = []
    cum_prob = 0.0
    
    for t in range(K+1):
        cum_prob += (1-p)**t
        
        thresholds.append(int(cum_prob / total_weight * M))
    
    return thresholds  # 长度为 K+1 的升序数组

def transform_uniform_to_time(u, thresholds):
    """
    transform uniform integer u to truncated geometric distribution
    """
    # 二分查找：找到第一个满足 u < thresholds[t] 的 t
    import bisect
    return bisect.bisect_right(thresholds, u)

class LifeGameDataset(Dataset):
    def __init__(self, file_list: List[str], sampling_distribution:str = "uniform"):
        """
        Args:
            file_list (List[str]): List of file paths to use in the dataset.
        """
        self.file_list = file_list
        self.arr_list = [np.load(i) for i in file_list]
        self.data_len_ls = [i.shape[0] for i in self.arr_list]
        self.len_prefix_sum = list(accumulate(self.data_len_ls))
        self.sampling_distribution = sampling_distribution
        
        self.transform_tables = {}
        
    def locate_idx(self, i: int) -> int:
        res_arr = [i-k for k in self.len_prefix_sum if k <= i]
        return len(res_arr) - 1
    
    def __len__(self) -> int:
        return self.len_prefix_sum[-1]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_idx = self.locate_idx(idx)
        file_path = self.file_list[file_idx]
        data = np.load(file_path)  # Shape: [T, N, N]
        
        # sample time w.r.t. truncated geometric(0.01).
        while (a := int(np.random.geometric(0.01)) - 1) >= data.shape[0] - 1:
            ...
        t = a
        
        x, x_1 = map(lambda x: torch.tensor(x, dtype=torch.float32), 
                          [data[t], data[t + 1]])
        x = (x / (dim:=(255 if x.max() > 100 else 1)))[None, ...]
        y = (x_1 / dim)
        
        #! Data Augmentation (deprecated)
        # x = transform(x)
        
        return x, y

def get_dataloader(data_dir, 
                   batch_size: int, 
                   shuffle: bool = True, 
                   num_workers: int = 0, 
                   split: Literal["train", "test", "all"] = "train"
                   ) -> DataLoader:
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