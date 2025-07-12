from functools import partial
from typing import Any, Callable, Iterable, Tuple
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
from torch.nn.functional import softmax, cross_entropy

from einops import rearrange, reduce
from jaxtyping import Float, Array

import os 

bimsa_life_100_dir = os.environ.get('BIMSA_LIFE_100_DIR')

# Assuming dataloader and model_conv are already defined
# Replace these with your actual imports or definitions

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to device
model = SimpleAutoencoder().to(device)

# Define loss function and optimizer
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
def train_model(model: Callable[[Float[Array, "batch 1 w h"]], 
                                Tuple[Float[Array, "batch 2 w h"], 
                                      Float[Array, "batch 1 w h"], 
                                      Float[Array, "batch h_dim"], 
                                      Float[Array, "batch h_dim"]]], 
                train_loader: Iterable[Tuple[Float[Array, "batch 1 w h"], Float[Array, "batch 1 w h"]]], 
                val_loader: Iterable[Tuple[Float[Array, "batch 1 w h"], Float[Array, "batch 1 w h"]]], 
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
        
        best_acc = 0.0
        
        for idx, (inputs, labels) in tqdm(enumerate(train_loader)):
            # Zero the parameter gradients
            optimizer.zero_grad()

            r_ratio = max(1e-3, r_ratio * (1 - 1e-6))  # Decrease r_ratio over epochs

            # Forward pass
            inputs_o, labels_o = inputs.clone(), labels.clone()
            inputs_t, labels_t = apply_translation(inputs_o), apply_translation(labels_o)
            inputs_r, labels_r = apply_rotation(inputs_o), apply_rotation(labels_o)
            
            x, y = torch.cat([inputs_o, inputs_t, inputs_r], dim=0), torch.cat([labels_o, labels_t, labels_r], dim=0)
            inputs, labels = x.to(device), y.to(device)
            
            outputs, r_inputs, hidden_a, hidden_b = model(inputs.to(device))
            
            # Dynamics Loss
            bs, ch, *_ = outputs.shape
            output_softmax = softmax(rearrange(outputs, "batch c w h -> (batch w h) c"), dim=-1)
            output_class_num = ([(dead_r:=(labels == 0).sum()), labels.numel() - dead_r])
            d_loss = cross_entropy(output_softmax, onehot_fn(labels.to(device).reshape(-1).long()).float(), weight=labels.numel() / torch.tensor(output_class_num, dtype=torch.float32).to(device))

            # Reconstruction Loss
            r_input_softmax = softmax(rearrange(r_inputs, "batch c w h -> (batch w h) c"), dim=-1)
            input_class_num = [(dead_r:=(inputs == 0).sum()), labels.numel() - dead_r]
            r_loss = cross_entropy(r_input_softmax, onehot_fn(inputs.to(device).reshape(-1).long()).float(), weight=labels.numel() / torch.tensor(input_class_num, dtype=torch.float32).to(device))
            
            # # Regularization Loss
            l_loss = torch.norm(hidden_a) + torch.norm(hidden_b)
            
            # Total Loss
            loss = (1 - r_ratio) * d_loss + r_ratio * r_loss + 1e-8 * l_loss
            wandb.log({"total_loss": loss.item(),
                       "dynamics_loss": d_loss.item(),
                       "reconstruction_loss": r_loss.item(),
                       "regularization_loss": l_loss.item()
                       })
            
            # Backward pass and optimization    
            loss.backward()
            
            # apply gradient clipping
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            wandb.log({
                       "gradient_norm": norm.item()
            })
            
            optimizer.step()

            # Statistics

                # _, predicted = outputs.argmax(1, keepdims=True)
                # val_total += labels.view(-1).size(0)
                # val_correct += predicted.eq(labels).sum().item()
                
            running_loss += loss.item()
            predicted: Float[Array, "batch 1 w h"] = outputs.argmax(1, keepdims=True)
            total += labels.view(-1).size(0)
            total += labels.numel()
            correct += item_correct:=predicted.eq(labels.to(device)).sum().item()
            
            wandb.log({"item_acc": item_correct / labels.numel() * 100})
            
            if idx % 100 == 0:
                    sample_idx = torch.randint(0, inputs.shape[0], (1,)).item()
                    
                    # construct image grid for wandb
                    img_output = outputs[sample_idx, 1][None, ...].cpu()
                    labels_output = labels[sample_idx].cpu()
                    predicted_output = predicted[sample_idx].cpu()
                    
                    wandb.log({
                        "train_sample": wandb.Image(rearrange([labels_output, img_output, predicted_output],
                                                              "n c h w -> c h (n w)").cpu()),
                    })
        
            assert correct <= total, f"Correct predictions {correct} exceed total {total}."
            

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            # Save the model checkpoint if needed
            torch.save(model.state_dict(), f'best_life_UNet_{SimpleAutoencoder.__version__}.pth')
        torch.save(model.state_dict(), f'last_life_UNet_{SimpleAutoencoder.__version__}.pth')
        
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%")
        
        wandb.log({"train_epoch_loss": epoch_loss, "train_epoch_acc": epoch_acc})
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            criterion = nn.CrossEntropyLoss() 
            for idx, (inputs, labels) in tqdm(enumerate(val_loader)):
                inputs, labels = inputs.to(device), labels.to(device)

                outputs, *_ = model(inputs)
                bs, ch, *_ = outputs.shape
                output_softmax = softmax(rearrange(outputs, "batch c w h -> (batch w h) c"), dim=-1)
                loss = criterion(output_softmax, 
                                 onehot_fn(labels.to(device).reshape(-1).long()).float())

                val_loss += loss.item()
                
                # TODO: Check.
                predicted: Float[Array, "batch 1 w h"] = outputs.argmax(1, keepdims=True)
                val_total += labels.numel()
                val_correct += predicted.eq(labels).sum().item()
        
                assert val_correct <= val_total, f"Validation correct predictions {val_correct} exceed total {val_total}."
                
                if idx % 100 == 0:
                        sample_idx = torch.randint(0, inputs.shape[0], (1,)).item()
                        
                        # construct image grid for wandb
                        img_output = outputs[sample_idx, 1][None, ...].cpu()
                        labels_output = labels[sample_idx].cpu()
                        predicted_output = predicted[sample_idx].cpu()
                        
                        wandb.log({
                            "test_sample": wandb.Image(rearrange([img_output, labels_output, predicted_output], 
                                                                "n c h w -> c h (n w)").cpu()),
                        })
        
        val_epoch_loss = val_loss / len(val_loader)
        val_epoch_acc = 100. * val_correct / val_total
        print(f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.2f}%")
        
        wandb.log({"val_epoch_loss": val_epoch_loss, "val_epoch_acc": val_epoch_acc})

train_loader = get_dataloader(
    data_dir=bimsa_life_100_dir,
    batch_size=8,
    shuffle=True,
    num_workers=0,
    split='train'
)

test_loader = get_dataloader(
    data_dir=bimsa_life_100_dir,
    batch_size=2,
    shuffle=False,
    num_workers=0,
    split='test'
)

if __name__ == "__main__":
    print("Starting training...")
    # Call the training function
    
    wandb.init(project="predictor_life")  # Replace with your WandB entity name
    
    train_model(model, train_loader, test_loader, None, optimizer, num_epochs=10)