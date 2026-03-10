import os
import re

import numpy as np
import torch
from einops import rearrange
from pathlib import Path

from dataloader import LifeGameDataset
import model_conv

def get_pth_list(folder_path):
    path = Path(folder_path)
    # 返回生成器，可以转为list
    pth_files = list(filter(lambda x: "old_history" not in str(x), path.rglob("*.pth")))
    
    # 按文件名排序
    pth_files.sort(key=lambda x: x.name)
    
    print(f"Found {len(pth_files)} .pth files\nthe first is {pth_files[0].name} at\n{pth_files[0]}")
    
    return pth_files  # 返回列表供后续使用

def process_erf(s):
    p = re.compile(r".*?/(?P<date>\d{4}(-\d\d){2})_(\d\d-){2}\d\d_.*?__(?P<n1>\d+)-(?P<n2>\d+)-(?P<rule>B\d*_S\d*V?)/best_simple_life_(?P<model>.*?)_\d.*?\.pth")
    match_dict = p.match(s).groupdict()
    rule_str, model_cls = match_dict["rule"], getattr(model_conv, match_dict["model"])

    d = f"./datasets/200-200-{rule_str}"

    print(rule_str, model_cls.__name__)

    model = model_cls()
    model.load_state_dict(torch.load(s, map_location=torch.device('cpu')))

    all_files = [os.path.join(d, f) for f in os.listdir(d) if f.endswith('.npy')]
    all_files.sort()  # Ensure deterministic split
    test_files = all_files[::5]
    file_list = [i for i in all_files if i not in test_files]

    dataset = LifeGameDataset(file_list=file_list)
    data = np.stack(dataset.arr_list, axis=1).reshape(-1, 1, 200, 200)

    input_original_tensors = torch.tensor(data).float().requires_grad_(False)

    print(input_original_tensors.shape)
    input_tensors = rearrange(torch.nn.Unfold(kernel_size=(11, 11), padding=0, stride=40)(input_original_tensors),
                            "n (a b) l -> (n l) 1 a b", a=11, b=11).detach().clone().requires_grad_(True)
    print(input_tensors.shape)

    out = model(input_tensors)
    *_, row, col = out.shape
    row, col
    grad = torch.zeros_like(out)
    grad[0, 0, row//2, col//2] = 1

    out.backward(gradient=grad)
    print(input_tensors.grad.shape)
    grad_abs = input_tensors.grad.abs().clone()

    res_max, res_avg = torch.max(grad_abs, dim=0)[0][0].numpy(), torch.mean(grad_abs, dim=(0, 1)).numpy()

    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(20, 8), dpi=300)
    plt.suptitle(f"Everage and Maximum Effective Respective Fields of {model.__class__.__name__} trained on B3678/S34678 Data")

    ax1 = plt.subplot(1, 2, 1)
    ax1.set_title(r"Maximum Absolute ERF accross 3200 samples")
    annot_matrix = np.where(np.abs(res_max) >= 0.01, np.round(res_max, 2), '')
    sns.heatmap(res_max, cmap="Blues", annot=annot_matrix, fmt='', ax=ax1, vmax=5, linewidths=1, linecolor="gray")
    ax1.set_xticks([]); ax1.set_yticks([])

    ax2 = plt.subplot(1, 2, 2)
    ax2.set_title(r"Average Absolute ERF accross 3200 samples")
    annot_matrix = np.where(np.abs(res_avg) >= 0.01, np.round(res_avg, 2), '')
    sns.heatmap(res_avg, cmap="Blues", annot=annot_matrix, fmt='', ax=ax2, vmax=5, linewidths=1, linecolor="gray")
    ax2.set_xticks([]); ax2.set_yticks([])


if __name__ == "__main__":
    pth_dir = input("Source .pth dir ::")
    
    pth_list = get_pth_list(pth_dir)
    
    for p in pth_list:
        print(f"Processing: {p}")
        process_erf(str(p))