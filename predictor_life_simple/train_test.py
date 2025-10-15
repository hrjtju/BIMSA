import argparse
import datetime
from functools import partial
from typing import Iterable, Tuple
import numpy as np
import toml
import torch
from torch import Tensor
from tqdm import tqdm
import wandb
from dataloader import get_dataloader
import model_conv
import matplotlib.pyplot as plt
from matplotlib import axes
from io import BytesIO
from PIL import Image
import pandas as pd
import warnings

from model_conv import GroupEquivariantCNN

import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import pad
from torch.nn.functional import softmax, cross_entropy
from torchinfo import summary

from einops import rearrange, reduce
from jaxtyping import Float, Array

import os 

warnings.filterwarnings("ignore")

# TODO: Update Dataset Name
bimsa_life_dir = os.environ.get('BIMSA_LIFE_DIR', "./predictor_life_simple/datasets")
os.environ['WANDB_BASE_URL'] = "https://api.bandw.top"
# os.path.append("./predictor_life/hyperparams/")

torch2numpy = lambda x: x[0].permute(1, 2, 0).clone().detach().cpu().numpy()

scalar_dict = {
    "train_loss": [],
    "train_acc": [],
    "grad_norm": [],
    "val_acc": [],
}

def save_image(inputs, labels, outputs, 
               idx: int, epoch: int, base_dir: str):
    
    if labels.shape[1] == 1:
        labels_ = labels.repeat(1, 2, 1, 1)
    elif len(labels.shape) == 3:
        labels_ = labels.unsqueeze(1)
    else:
        labels_ = labels
    
    random_index = np.random.randint(0, inputs.shape[0])
    
    x_t0: Float[Array, "w h"] = inputs[random_index, 0].cpu() * 256
    x_t1: Float[Array, "w h"] = inputs[random_index, 1].cpu() * 256
    y_t1: Float[Array, "w h"] = labels_[random_index, 0].cpu() * 256
    xp_t0: Float[Array, "w h"] = outputs[random_index, 0].cpu() * 256

    
    fig, ((ax1, ax2, ax5), (ax3, ax4, ax6)) = plt.subplots(2, 3, figsize=(12, 8), dpi=200)
    
    ax1: axes.Axes
    ax2: axes.Axes
    ax3: axes.Axes
    ax4: axes.Axes
    ax5: axes.Axes
    ax6: axes.Axes
    
    ax1.axis("off")
    ax1.imshow(x_t0, cmap='gray')
    ax1.set_title("$x_{t}$\nTrue System State at time t")
    ax2.axis("off")
    ax2.imshow(x_t1, cmap='gray')
    ax2.set_title("$x_{t+1}$\nTrue System State at time t+1")
    ax4.axis("off")
    ax4.imshow(xp_t0, cmap='gray')
    ax4.set_title("$\hat{x}_{t+2} = f(x_{t+1})$\nPredicted State at time t+2")

    ax3.axis("off")
    ax3.imshow(y_t1, cmap='gray')
    ax3.set_title("$x_{t+2}$\nTrue System State at time t+2")
    ax5.axis("off")
    ax5.imshow(y_t1 - xp_t0, cmap="RdBu", vmin=-1, vmax=1)
    ax5.set_title("$x_{t+1} - f(x_{t+1})$\nPrediction Error")
    if torch.norm(y_t1 - xp_t0) < 1e-6:
        # display "ALL CORRECT" in the center of the figure
        ax5.text(0.5, 0.5, "ALL CORRECT", 
                 fontsize=20, color='green', ha='center', va='center', transform=ax5.transAxes
                 )
    
    ax6.axis("off")
    ax6.imshow(x_t1 - x_t0, cmap="RdBu", vmin=-1, vmax=1)
    ax6.set_title("$x_{t+1} - x_{t}$\nDifference between $x_{t+1}$ and $x_{t}$")

    # convert plotting results to array format
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    image = np.array(Image.open(buf))[:, :, :3]
    
    # save plotting results
    if not os.path.exists(f"result/predictor_life_simple/{base_dir}"):
        os.makedirs(f"result/predictor_life_simple/{base_dir}")
    plt.savefig(fig_f:=f"result/predictor_life_simple/{base_dir}/train_sample_{epoch:>02d}_{idx:>05d}.png", bbox_inches="tight")
    plt.close()
    
    with open(f"result/predictor_life_simple/{base_dir}/visualization.md", 'a') as f:
        f.write(f"\n![]({fig_f})\n<center>Iteration {idx+1}</center>\n")
    
    return image

def plot_scalar(scalar_dict: dict, base_dir: str) -> None:
    
    # plot the scalar_dict values and store the figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    ax1: axes.Axes
    ax2: axes.Axes
    ax3: axes.Axes
    ax4: axes.Axes
    
    ax1.plot(scalar_dict["train_loss"], label="train_loss", alpha=0.3)
    ax1.plot(l:=pd.Series(scalar_dict["train_loss"]).rolling(window=len(scalar_dict["train_loss"])//10, min_periods=1, center=True).mean(), label="train_loss (smoothed)", color="#1f77b4")
    ax1.legend()
    ax1.grid()
    ax1.set_ylim(0, max(l)*1.1)
    ax1.set_title("train_loss")

    ax2.plot(scalar_dict["grad_norm"], label="grad_norm", alpha=0.3)
    ax2.plot(pd.Series(scalar_dict["grad_norm"]).rolling(window=len(scalar_dict["grad_norm"])//10, min_periods=1, center=True).mean(), label="grad_norm (smoothed)", color="#1f77b4")
    ax2.legend()
    ax2.semilogy()
    ax2.grid()
    ax2.set_title("grad_norm")

    ax3.plot(scalar_dict["train_acc"], label="train_acc", alpha=0.3)
    ax3.plot(pd.Series(scalar_dict["train_acc"]).rolling(window=len(scalar_dict["train_acc"])//10, min_periods=1, center=True).mean(), label="train_acc (smoothed)", color="#1f77b4")
    ax3.legend()
    ax3.grid()
    ax3.set_title("train_acc")
    
    ax4.plot(scalar_dict["val_acc"], label="val_acc")
    ax4.set_title("val_acc")
    ax4.grid()
    ax4.legend()

    # TODO: Update Dataset Name
    fig.savefig(f"result/predictor_life_simple/{base_dir}/scalar_dict.png")
    
    # save plotting results
    if not os.path.exists(f"result/predictor_life_simple/{base_dir}"):
        os.makedirs(f"result/predictor_life_simple/{base_dir}")
    
    plt.close()

def plot_network_analysis(model: nn.Module):
    """
    Plot the ALL trained weights of the CNN model.
    """
    
    # TODO: Update Dataset Name

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

def show_image_grid(inputs: Tensor|Float[Array, "batch 2 w h"], 
                    labels: Tensor|Float[Array, "batch 2 w h"], 
                    outputs: Tensor|Float[Array, "batch 2 w h"]) -> Tensor:

    # print(labels.shape, outputs.shape)
    
    if labels.shape[1] == 1:
        labels_ = labels.repeat(1, 2, 1, 1)
    elif len(labels.shape) == 3:
        labels_ = labels.unsqueeze(1)
    else:
        labels_ = labels
    
    # print(labels_.shape, outputs.shape)
    
    random_index = np.random.randint(0, inputs.shape[0])
    
    x_t0: Float[Array, "w h"] = pad(inputs[random_index, 0].cpu() * 256, (2, 2, 2, 2), value=128)
    x_t1: Float[Array, "w h"] = pad(inputs[random_index, 1].cpu() * 256, (2, 2, 2, 2), value=128)
    y_t1: Float[Array, "w h"] = pad(labels_[random_index, 0].cpu() * 256, (2, 2, 2, 2), value=128)
    xp_t0: Float[Array, "w h"] = pad(outputs[random_index, 0].cpu() * 256, (2, 2, 2, 2), value=128)

    image_grid = rearrange([x_t0, xp_t0, y_t1, x_t1],
                           "(b1 b2) w h -> 1 (b1 b2 w) h",
                           b1=2, b2=2
                           ).cpu()
    
    return image_grid
                

# Training function
def train_model(
                args: dict = None
                ):
    
    rule_data_str = f"{args['sys_size']}-{args['data_iters']}-{args['data_rule'].replace('/', '_')}"
    dataset_dir = f"{bimsa_life_dir}/{rule_data_str}/"
    start_time_str = f"{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}".split('.')[0]
    save_base_str = f"{start_time_str}_{args['wandb']['entity']}__{rule_data_str}"
    
    print(f"\nPicking Dataset: {dataset_dir}\nSaving Base Directory: {save_base_str}\n\n")
    
    if args["wandb"]["turn_on"]:
        wandb.init(project="predictor_life_simple", name=args["wandb"]["entity"])
    else:
        wandb.init(mode="disabled")
    
    # TODO: Update Dataset Name
    train_loader: Iterable[Tuple[Float[Array, "batch 2 w h"], Float[Array, "batch 2 w h"]]] = get_dataloader(
        data_dir=dataset_dir,
        batch_size=args["dataloader"]["train_batch_size"],
        shuffle=args["dataloader"]["train_shuffle"],
        num_workers=args["dataloader"]["train_num_workers"],
        split='train'
    )

    # TODO: Update Dataset Name
    test_loader: Iterable[Tuple[Float[Array, "batch 2 w h"], Float[Array, "batch 2 w h"]]] = get_dataloader(
        data_dir=dataset_dir,
        batch_size=args["dataloader"]["test_batch_size"],
        shuffle=args["dataloader"]["test_shuffle"],
        num_workers=args["dataloader"]["test_num_workers"],
        split='test'
    )
        
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_class = getattr(model_conv, args["model"]["name"])
    
    # Move model to device
    model: nn.Module|GroupEquivariantCNN = model_class()

    try:
        summary(model, input_size=(1, 2, args['sys_size'], args['sys_size']), verbose=1)
    except:
        summary(model.cpu().export(), input_size=(1, 2, args['sys_size'], args['sys_size']), verbose=1)
    
    model = model.to(device)
    
    # model = torch.compile(model)
    
    # Define loss function and optimizer
    optimizer = getattr(optim, args["optimizer"]["name"])(model.parameters(), **args["optimizer"]["args"])

    onehot_fn = partial(torch.nn.functional.one_hot, num_classes=2)
    one_hot_target = lambda x: onehot_fn(x.reshape(-1).long()).float()
    
    # TODO: 需要将模型训练切分为两个阶段：第一阶段训练 Encoder 和 Decoder，缩小重构损失
    # TODO: 第二阶段训练 Dynamics 模块，缩小动力学损失
    
    match args["lr_scheduler"]["name"]:
        case "None":
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
        
        for idx, (inputs, labels) in enumerate(train_loader):
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            inputs_o, labels_o = inputs.clone(), labels.clone()
            inputs_t, labels_t = apply_translation(inputs_o, labels_o)
            inputs_r, labels_r = apply_rotation(inputs_o, labels_o)
            
            # Concatenate original, translated, and rotated inputs and labels
            x, y = torch.cat([inputs_o, inputs_t, inputs_r], dim=0), torch.cat([labels_o, labels_t, labels_r], dim=0)
            inputs: Float[Array, "batch 2 w h"] = x.to(device)
            labels: Float[Array, "batch 1 w h"] = y.to(device)
            
            outputs = model(inputs.to(device))
            outputs_logits = rearrange(outputs, "b c w h -> (b w h) c")
            
            #TODO: add L1 loss
            l1_reg = 0
            for name, param in model.named_parameters():
                if 'weight' in name:
                    l1_reg = l1_reg + torch.linalg.vector_norm(param, ord=1, dim=None)
            
            # Dynamics Loss
            output_class_num = ([(dead_r:=(labels == 0).sum()), labels.numel() - dead_r])
            d_loss = cross_entropy(outputs_logits, one_hot_target(labels.to(device)), weight=labels.numel() \
                / torch.tensor(output_class_num, dtype=torch.float32).to(device)) + 1e-5 * l1_reg

            d_loss.backward()
            
            # apply gradient clipping
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            if use_lr_scheduler:
                scheduler.step()
                
            running_loss += d_loss.item()
            predicted: Float[Array, "batch 1 w h"] = (outputs.argmax(dim=1)).long()
            total += labels.numel()
            
            # print(f"Predicted: {predicted.shape}, Labels: {labels.shape}")
            
            correct += (item_correct:=predicted.eq(labels.to(device)).sum().item())
            
            wandb.log({"total_loss": d_loss.item(),
                       "gradient_norm": norm.item(),
                       "item_acc": (item_acc:=(item_correct / labels.numel() * 100))
                       })
            scalar_dict["train_loss"].append(d_loss.item())
            scalar_dict["train_acc"].append(item_acc)
            scalar_dict["grad_norm"].append(norm.item())
            
            if idx % 100 == 0:
                
                # 将 image_grid 异步存储为 matplotlib 图像
                image_grid = save_image(inputs, labels, outputs.argmax(dim=1)[:, None, ...],
                           idx, epoch,
                           save_base_str
                          )
                
                wandb.log({
                    "train_sample": wandb.Image(image_grid),
                })

            assert correct <= total, f"Correct predictions {correct} exceed total {total}."
            
            if (idx+1) % 40 == 0:
                print(f"| {datetime.datetime.now()} | Idx: {idx+1:>4d}/{len(train_loader):<6d} "
                      f"| loss: {running_loss/(idx+1):.3f} "
                      f"| grad_norm: {norm:.3f} | acc: {item_acc:.2f}% |")
        

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
            
        # TODO: Update Dataset Name
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            # Save the model checkpoint if needed
            torch.save(model.state_dict(), f"./result/predictor_life_simple/{save_base_str}/"
                       f"best_simple_life_{model_class.__name__}_{model_class.__version__}.pth")
        torch.save(model.state_dict(), f"./result/predictor_life_simple/{save_base_str}/"
                   f"last_simple_life_{model_class.__name__}_{model_class.__version__}.pth")

        
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%", flush=True)
        
        wandb.log({"train_epoch_loss": epoch_loss, "train_epoch_acc": epoch_acc})
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for idx, (inputs, labels) in tqdm(enumerate(test_loader)):
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                
                # TODO: Check.
                predicted: torch.Tensor|Float[Array, "batch 1 w h"] = (outputs.argmax(dim=1)).long()
                val_total += labels.numel()
                val_correct += predicted.eq(labels).sum().item()
        
                assert val_correct <= val_total, f"Validation correct predictions {val_correct} exceed total {val_total}."
                
                if idx % 100 == 0:
                    
                        image_grid = save_image(inputs, labels, outputs.argmax(dim=1)[:, None, ...],
                           idx, epoch,
                           save_base_str
                          )
                        
                        wandb.log({
                            "test_sample": wandb.Image(image_grid),
                        })
        
        val_epoch_acc = 100. * val_correct / val_total
        print(f"Acc: {val_epoch_acc:.2f}%", flush=True)
        scalar_dict["val_acc"].append(val_epoch_acc)
        
        wandb.log({"val_epoch_acc": val_epoch_acc})
        
        if (val_epoch_acc > 99 
                and epoch > 3 
                and np.all(scalar_dict["train_loss"][-30:] < 1.5e-2)
                and np.mean(scalar_dict["train_acc"][-30:]) > 99
                ):
            print("Early Stopped.")
            break
    
    plot_scalar(scalar_dict, save_base_str)
    torch.cuda.empty_cache()
    # plot_network_analysis(model, f"{start_time_str}_{args['wandb']['entity']}")


if __name__ == "__main__":
    # reads the command line arguments
    in_profile = argparse.ArgumentParser(description="Train the Predictor Life model")
    
    in_profile.add_argument("-p", "--hyperparameters", type=str, default="./predictor_life_simple/hyperparams/baseline.toml", help="Path to hyperparameters file")
    in_profile.add_argument("-r", "--sysRule", dest="data_rule", type=str, default="B3/S23", help="Life rules")
    in_profile.add_argument("-i", "--dataIter", dest="data_iters", type=int, default=200, help="Iterations within each data file")
    in_profile.add_argument("-w", "--sysSize", dest="sys_size", type=int, default=200, help="System size")
    
    in_profile_args = in_profile.parse_args()

    print(in_profile_args.hyperparameters, end='\n\n')
    args_dict = toml.load(in_profile_args.hyperparameters) | dict(in_profile_args._get_kwargs())
    print(args_dict)
    print("Starting training...")
    # Call the training function
    
    train_model(args_dict)