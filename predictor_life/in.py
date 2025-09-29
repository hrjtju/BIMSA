from functools import partial
from typing import Any, Callable, Tuple
import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm
import wandb
from dataloader import get_dataloader
from model_conv import SimpleAutoencoder

import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import os 
BIMSA_LIFE_DIR = os.environ.get('BIMSA_LIFE_DIR', '/root/autodl-tmp/life/')

# Assuming dataloader and model_conv are already defined
# Replace these with your actual imports or definitions

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to device
model = SimpleAutoencoder().to(device)

# Define loss function and optimizer
criterion = nn.BCELoss()  # Replace with appropriate loss for your task
optimizer = optim.Adam(model.parameters(), lr=1e-5)

def apply_translation(grid: Tensor) -> Tensor:
    """
    Applies a spatial translation to the grid
    
    Args:
        grid (Tensor): The input grid.
    
    Returns:
        Tensor: Translated grid.
    """
    N = grid.shape[-1]
    i, j = np.random.randint(0, N, size=2)
    translated_grid = torch.roll(grid, shifts=(-i, -j), dims=(-2, -1))
    return translated_grid

def apply_rotation(grid: Tensor) -> Tensor:
    """
    Applies a rotation to the grid.
    
    Args:
        grid (Tensor): The input grid.
        angle (int): Rotation angle in degrees (must be 90, 180, or 270).
    
    Returns:
        Tensor: Rotated grid.
    """
    
    angle = np.random.choice([90, 180, 270])  # Randomly choose an angle
    
    # Rotate the grid using PyTorch's tensor operations
    if angle == 90:
        rotated_grid = grid.transpose(-2, -1).flip(-1)
    elif angle == 180:
        rotated_grid = grid.flip(-2).flip(-1)
    elif angle == 270:
        rotated_grid = grid.transpose(-2, -1).flip(-2)
    
    return rotated_grid

# Training function
def train_model(model: Callable[[Tensor], Tuple[Tensor, Tensor, Tensor, Tensor]], 
                train_loader: DataLoader, 
                val_loader: DataLoader, 
                criterion: Callable[[Any], Tensor], 
                optimizer: Optimizer, 
                num_epochs: int = 10
                ):
    
    onehot_fn = partial(torch.nn.functional.one_hot, num_classes=2)
    
    # TODO: 需要将模型训练切分为两个阶段：第一阶段训练 Encoder 和 Decoder，缩小重构损失
    # TODO: 第二阶段训练 Dynamics 模块，缩小动力学损失
    
    r_ratio = 1
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        
        for inputs, labels in tqdm(train_loader):
            # Zero the parameter gradients
            optimizer.zero_grad()

            r_ratio = max(1e-3, r_ratio * (1 - 1e-6))  # Decrease r_ratio over epochs

            # Forward pass
            # inputs_o, labels_o = inputs.clone(), labels.clone()
            # inputs_t, labels_t = apply_translation(inputs_o), apply_translation(labels_o)
            # inputs_r, labels_r = apply_rotation(inputs_o), apply_rotation(labels_o)
            
            # x, y = torch.cat([inputs_o, inputs_t, inputs_r], dim=0), torch.cat([labels_o, labels_t, labels_r], dim=0)
            # inputs, labels = x.to(device), y.to(device)
            
            outputs, r_inputs, hidden_a, hidden_b = model(inputs.to(device))
            
            # Dynamics Loss
            bs, ch, *_ = outputs.shape
            d_loss = criterion(outputs.reshape(-1, ch), 
                               onehot_fn(labels.to(device).reshape(-1).long()).float())

            # Reconstruction Loss
            r_loss = criterion(r_inputs.reshape(-1, ch), 
                               onehot_fn(inputs.to(device).reshape(-1).long()).float())
            
            # # Regularization Loss
            l_loss = torch.norm(hidden_a) + torch.norm(hidden_b)
            
            # Total Loss
            loss = (1 - r_ratio) * d_loss + r_ratio * r_loss + 1e-8 * l_loss
            
            # Backward pass and optimization    
            loss.backward()
            
            # apply gradient clipping
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            wandb.log({"total_loss": loss.item(),
                       "dynamics_loss": d_loss.item(),
                       "gradient_norm": norm.item(),
                       "reconstruction_loss": r_loss.item(),
                       "regularization_loss": l_loss.item()
                       })
            
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0) * labels.size(1) * labels.size(2)  # Assuming labels are 3D tensors
            correct += predicted.eq(labels.to(device)).sum().item()
            

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%")
        
        wandb.log({"train_epoch_loss": epoch_loss, "train_epoch_acc": epoch_acc})
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs, *_ = model(inputs)
                bs, ch, *_ = outputs.shape
                loss = criterion(outputs.reshape(-1, ch), 
                                 onehot_fn(labels.to(device).reshape(-1).long()).float())

                val_loss += loss.item()
                
                # TODO: Check.
                _, predicted = outputs.argmax(1, keepdims=True)
                val_total += labels.view(-1).size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_epoch_loss = val_loss / len(val_loader)
        val_epoch_acc = 100. * val_correct / val_total
        print(f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.2f}%")
        
        wandb.log({"val_epoch_loss": val_epoch_loss, "val_epoch_acc": val_epoch_acc})

train_loader = get_dataloader(
    data_dir=BIMSA_LIFE_DIR,
    batch_size=8,
    shuffle=True,
    num_workers=0,
    split='train'
)

test_loader = get_dataloader(
    data_dir=BIMSA_LIFE_DIR,
    batch_size=2,
    shuffle=False,
    num_workers=0,
    split='test'
)

if __name__ == "__main__":
    print("Starting training...")
    # Call the training function
    
    wandb.init(project="predictor_life")  # Replace with your WandB entity name
    
    train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10)