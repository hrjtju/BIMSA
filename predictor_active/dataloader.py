import os
import random
from typing import Literal, Tuple, List
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class LifeGameDataset(Dataset):
    
    def __init__(self, data_dir: str, split: Literal["train", "test", "all"] = "train"):
        """
        Args:
            file_list (List[str]): List of file paths to use in the dataset.
        """
        
        all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')]
        all_files.sort()  # Ensure deterministic split
        test_files = all_files[::5]
        
        self.split = split
        
        if split == "train":
            self.file_list = [i for i in all_files if i not in test_files]
            self.visible_trajectories = [random.choice(self.file_list)]
        elif split == "test":
            self.file_list = test_files
            self.visible_trajectories = self.file_list 
        elif split == "all":
            self.file_list = all_files
            self.visible_trajectories = self.file_list
        else:
            raise NotImplementedError(f"Argument 'split'(current value {split}) only accepts 'train', 'test' or 'all'.")
        
        self.len_per_file = np.load(self.visible_trajectories[0]).shape[0]

    def add_trajectory(self):
        if self.split == "train":
            self.visible_trajectories.append(random.choice(list(set(self.file_list)-set(self.visible_trajectories))))
    
    def __len__(self) -> int:
        return len(self.visible_trajectories) * self.len_per_file

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_idx = idx // self.len_per_file
        file_path = self.visible_trajectories[file_idx]
        data = np.load(file_path)  # Shape: [T, N, N]
        t = np.random.randint(0, data.shape[0] - 2)
        x, x_1 = torch.tensor(data[t], dtype=torch.float32), torch.tensor(data[t + 1], dtype=torch.float32)
        x = (x / (dim:=(255 if x.max() > 100 else 1)))[None, ...]
        y = (x_1 / dim)
        
        return x, y

    def get_dataloader(self, 
                        batch_size: int, 
                        shuffle: bool = True, 
                        num_workers: int = 0, 
                        
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
        
        dataloader = DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        
        return dataloader