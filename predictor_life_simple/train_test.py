import argparse
from functools import partial
from typing import Any, Callable, Iterable, Tuple
import numpy as np
import toml
import torch
from torch import Tensor
from tqdm import tqdm
import wandb
from dataloader import get_dataloader
from model_conv import SimpleCNN
from torchvision.utils import make_grid


import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import pad
from torch.nn.functional import softmax, cross_entropy

from einops import rearrange, reduce
from jaxtyping import Float, Array

import os 

bimsa_life_100_dir = os.environ.get('BIMSA_LIFE_100_DIR', "./predictor_life/datasets/life/")
# os.path.append("./predictor_life/hyperparams/")

# Assuming dataloader and model_conv are already defined
# Replace these with your actual imports or definitions

def apply_translation(*grids: Tuple[Tensor]) -> Tuple[Tensor]:
    """
    Applies a spatial translation to the grid
    
    Args:
        grid (Tensor): The input grid.
    
    Returns:
        Tensor: Translated grid.
    """
    result = []
    
    N = grids[0].shape[-1]
    i, j = np.random.randint(0, N, size=2)
    
    for grid in grids:
        translated_grid = torch.roll(grid, shifts=(-i, -j), dims=(-2, -1))
        result.append(translated_grid)
        
    return tuple(result)

def apply_rotation(*grids: Tuple[Tensor]) -> Tuple[Tensor]:
    """
    Applies a rotation to the grid.
    
    Args:
        grid (Tensor): The input grid.
        angle (int): Rotation angle in degrees (must be 90, 180, or 270).
    
    Returns:
        Tensor: Rotated grid.
    """
    result = []
    angle = np.random.choice([0, 90, 180, 270])  # Randomly choose an angle
    
    for grid in grids:
        # Rotate the grid using PyTorch's tensor operations
        if angle == 0:
            rotated_grid = grid
        elif angle == 90:
            rotated_grid = grid.transpose(-2, -1).flip(-1)
        elif angle == 180:
            rotated_grid = grid.flip(-2).flip(-1)
        elif angle == 270:
            rotated_grid = grid.transpose(-2, -1).flip(-2)
        result.append(rotated_grid)
        
    return tuple(result)

def show_image_grid(inputs: Float[Array, "batch 2 w h"], labels: Float[Array, "batch 2 w h"], outputs: Tensor) -> Tensor:
    
    random_index = np.random.randint(0, inputs.shape[0])
    
    x_t0: Float[Array, "w h"] = pad(inputs[random_index, 0].cpu() * 255, (2, 2, 2, 2), value=128)
    x_t1: Float[Array, "w h"] = pad(inputs[random_index, 1].cpu() * 255, (2, 2, 2, 2), value=128)
    y_t1: Float[Array, "w h"] = pad(labels[random_index, 1].cpu() * 255, (2, 2, 2, 2), value=128)
    y_t2: Float[Array, "w h"] = pad(labels[random_index, 1].cpu() * 255, (2, 2, 2, 2), value=128)
    xp_t0: Float[Array, "w h"] = pad(outputs[random_index, 0].cpu() * 255, (2, 2, 2, 2), value=128)
    xp_t1: Float[Array, "w h"] = pad(outputs[random_index, 1].cpu() * 255, (2, 2, 2, 2), value=128)
    
    image_grid = rearrange([x_t0, x_t1, y_t1, y_t2, xp_t0, xp_t1], 
                           "(b1 b2) w h -> 1 (b1 w) (b2 h)",
                           b1 = 2, b2 = 3
                           ).cpu()
    
    return image_grid
                

# Training function
def train_model(
                args: dict = None
                ):
    
    train_loader: Iterable[Tuple[Float[Array, "batch 2 w h"], Float[Array, "batch 2 w h"]]] = get_dataloader(
        data_dir=bimsa_life_100_dir,
        batch_size=args["dataloader"]["train_batch_size"],
        shuffle=args["dataloader"]["train_shuffle"],
        num_workers=args["dataloader"]["train_num_workers"],
        split='train'
    )

    test_loader: Iterable[Tuple[Float[Array, "batch 2 w h"], Float[Array, "batch 2 w h"]]] = get_dataloader(
        data_dir=bimsa_life_100_dir,
        batch_size=args["dataloader"]["test_batch_size"],
        shuffle=args["dataloader"]["test_shuffle"],
        num_workers=args["dataloader"]["test_num_workers"],
        split='test'
    )
    
        
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_class = SimpleCNN
    
    # Move model to device
    model = model_class().to(device)

    # Define loss function and optimizer
    optimizer = getattr(optim, args["optimizer"]["name"])(model.parameters(), **args["optimizer"]["args"])

    onehot_fn = partial(torch.nn.functional.one_hot, num_classes=2)
    one_hot_target = lambda x: onehot_fn(x.reshape(-1).long()).float()
    
    # TODO: 需要将模型训练切分为两个阶段：第一阶段训练 Encoder 和 Decoder，缩小重构损失
    # TODO: 第二阶段训练 Dynamics 模块，缩小动力学损失
    
    match args["lr_scheduler"]["name"]:
        case None:
            use_lr_scheduler = False
        case _:
            use_lr_scheduler = True
            scheduler = getattr(optim.lr_scheduler, args["lr_scheduler"]["name"])(optimizer, **args["lr_scheduler"]["args"])

    for epoch in range(epochs:=args["training"]["epochs"]):
        print(f"Epoch {epoch+1}/{epochs}")
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

            # Forward pass
            inputs_o, labels_o = inputs.clone(), labels.clone()
            inputs_t, labels_t = apply_translation(inputs_o, labels_o)
            inputs_r, labels_r = apply_rotation(inputs_o, labels_o)
            
            # Concatenate original, translated, and rotated inputs and labels
            x, y = torch.cat([inputs_o, inputs_t, inputs_r], dim=0), torch.cat([labels_o, labels_t, labels_r], dim=0)
            inputs, labels = x.to(device), y.to(device)
            
            outputs, n_output = model(inputs.to(device))
            
            output_f, n_output_f = outputs.reshape(-1), n_output.reshape(-1)
            output_t = torch.stack([output_f, n_output_f], dim=-1)
            
            # Dynamics Loss
            output_class_num = ([(dead_r:=(labels == 0).sum()), labels.numel() - dead_r])
            d_loss = cross_entropy(output_t, one_hot_target(labels.to(device)), weight=labels.numel() \
                / torch.tensor(output_class_num, dtype=torch.float32).to(device))
            
            wandb.log({"total_loss": d_loss.item(),
                       })

            d_loss.backward()
            
            # apply gradient clipping
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            wandb.log({
                       "gradient_norm": norm.item()
            })
            
            optimizer.step()
            if use_lr_scheduler:
                scheduler.step()
                
            running_loss += d_loss.item()
            predicted: Float[Array, "batch 1 w h"] = outputs.argmax(1, keepdims=True)
            total += labels.numel()
            correct += (item_correct:=predicted.eq(labels.to(device)).sum().item())
            
            wandb.log({"item_acc": item_correct / labels.numel() * 100})
            
            if idx % 100 == 0:
                
                image_grid = show_image_grid(inputs, labels, outputs)
                
                wandb.log({
                    "train_sample": wandb.Image(image_grid),
                })
        
            assert correct <= total, f"Correct predictions {correct} exceed total {total}."
            

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            # Save the model checkpoint if needed
            torch.save(model.state_dict(), f'best_life_UNet_{model_class.__version__}.pth')
        torch.save(model.state_dict(), f'last_life_UNet_{model_class.__version__}.pth')
        
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%")
        
        wandb.log({"train_epoch_loss": epoch_loss, "train_epoch_acc": epoch_acc})
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            criterion = nn.CrossEntropyLoss() 
            for idx, (inputs, labels) in tqdm(enumerate(test_loader)):
                inputs, labels = inputs.to(device), labels.to(device)

                outputs, n_output = model(inputs)
                
                output_f, n_output_f = outputs.reshape(-1), n_output.reshape(-1)
                output_t = torch.stack([output_f, n_output_f], dim=-1)
                
                bs, ch, *_ = outputs.shape
                loss = criterion(output_t, 
                                 onehot_fn(labels.to(device).reshape(-1).long()).float())

                val_loss += loss.item()
                
                # TODO: Check.
                predicted: Float[Array, "batch 1 w h"] = outputs.argmax(1, keepdims=True)
                val_total += labels.numel()
                val_correct += predicted.eq(labels).sum().item()
        
                assert val_correct <= val_total, f"Validation correct predictions {val_correct} exceed total {val_total}."
                
                if idx % 100 == 0:
                    
                        image_grid = show_image_grid(inputs, labels, outputs)
                        
                        wandb.log({
                            "test_sample": wandb.Image(image_grid),
                        })
        
        val_epoch_loss = val_loss / len(test_loader)
        val_epoch_acc = 100. * val_correct / val_total
        print(f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.2f}%")
        
        wandb.log({"val_epoch_loss": val_epoch_loss, "val_epoch_acc": val_epoch_acc})

if __name__ == "__main__":
    # reads the command line arguments
    in_profile = argparse.ArgumentParser(description="Train the Predictor Life model")
    in_profile.add_argument("-p", "--hyperparameters", type=str, default="./predictor_life/hyperparams/baseline.toml", help="Path to hyperparameters file")
    in_profile_args = in_profile.parse_args()

    args_dict = toml.load(in_profile_args.hyperparameters)

    print("Starting training...")
    # Call the training function
    
    if args_dict["wandb"]["turn_on"]:
        wandb.init(project="predictor_life", name=args_dict["wandb"]["entity"])  # Replace with your WandB entity name
        # wandb.init(project="predictor_life", mode="disabled")
    else:
        wandb.init(mode="disabled")

    
    train_model(args_dict)