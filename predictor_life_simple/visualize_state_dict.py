"""
可视化 state_dict 中的所有参数
支持从文件或直接从模型加载
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import re


def parse_state_dict_text(file_path: str) -> OrderedDict:
    """
    从文本文件中解析 state_dict
    适用于从 distillation 输出的文本复制保存的情况
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    state_dict = OrderedDict()
    
    # 使用正则表达式提取参数
    # 匹配模式: ('name', tensor([...]))
    pattern = r"\('([^']+)'\,\s*tensor\((\[.*?\])\s*\)\)"
    
    # 由于文本可能很大，我们需要逐行或分段解析
    # 这里采用更简单的策略：手动解析
    lines = content.split('\n')
    current_key = None
    current_tensor_str = []
    in_tensor = False
    
    for line in lines:
        line = line.strip()
        
        # 检测新的参数开始
        if line.startswith("('") and "', tensor(" in line:
            # 保存之前的参数
            if current_key and current_tensor_str:
                tensor_str = '\n'.join(current_tensor_str)
                try:
                    # 解析 tensor 字符串
                    tensor = parse_tensor_string(tensor_str)
                    if tensor is not None:
                        state_dict[current_key] = tensor
                except Exception as e:
                    print(f"解析 {current_key} 失败: {e}")
            
            # 提取新参数的 key
            match = re.match(r"\('([^']+)'\,\s*tensor\(", line)
            if match:
                current_key = match.group(1)
                # 提取 tensor 内容开始
                tensor_start = line.find("tensor(") + 7
                current_tensor_str = [line[tensor_start:]]
                in_tensor = True
        
        elif in_tensor:
            current_tensor_str.append(line)
            if line.rstrip().endswith('))'):
                # tensor 结束
                tensor_str = '\n'.join(current_tensor_str)
                try:
                    tensor = parse_tensor_string(tensor_str)
                    if tensor is not None:
                        state_dict[current_key] = tensor
                except Exception as e:
                    print(f"解析 {current_key} 失败: {e}")
                current_key = None
                current_tensor_str = []
                in_tensor = False
    
    # 处理最后一个参数
    if current_key and current_tensor_str:
        tensor_str = '\n'.join(current_tensor_str)
        try:
            tensor = parse_tensor_string(tensor_str)
            if tensor is not None:
                state_dict[current_key] = tensor
        except Exception as e:
            print(f"解析 {current_key} 失败: {e}")
    
    return state_dict


def parse_tensor_string(s: str) -> torch.Tensor:
    """
    将 tensor 的字符串表示解析为实际的 tensor
    支持嵌套的多维数组格式
    """
    # 清理字符串
    s = s.strip()
    if s.endswith(')'):
        s = s[:-1]
    if s.endswith(']'):
        s = s[:-1]
    
    try:
        # 使用 numpy 解析数组
        # 先将 PyTorch tensor 格式转换为 numpy 可读的格式
        s = s.replace('tensor(', '').replace(')', '')
        
        # 使用 eval 解析（在受控环境下安全）
        import ast
        array = ast.literal_eval(s)
        
        return torch.tensor(array, dtype=torch.float32)
    except Exception as e:
        # 如果解析失败，返回 None
        return None


def load_state_dict(file_path: str) -> OrderedDict:
    """
    加载 state_dict，支持 .pth 文件和 .txt 文本文件
    """
    if file_path.endswith('.pth') or file_path.endswith('.pt'):
        return torch.load(file_path, map_location='cpu')
    else:
        # 假设是文本文件
        return parse_state_dict_text(file_path)


def visualize_fusion_weight(state_dict: OrderedDict, save_path: str = "fusion_weight.png"):
    """可视化 fusion_weight"""
    if 'fusion_weight' not in state_dict:
        print("fusion_weight not found")
        return
    
    fusion_weight = state_dict['fusion_weight']
    if isinstance(fusion_weight, torch.Tensor):
        fusion_weight = fusion_weight.cpu().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 条形图
    axes[0].bar(range(len(fusion_weight)), fusion_weight, color='steelblue', edgecolor='black')
    axes[0].set_xlabel('Head Index')
    axes[0].set_ylabel('Weight Value')
    axes[0].set_title('Fusion Weights (8 Heads)')
    axes[0].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[0].grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for i, v in enumerate(fusion_weight):
        axes[0].text(i, v, f'{v:.3f}', ha='center', 
                    va='bottom' if v > 0 else 'top', fontsize=9)
    
    # 热力图
    im = axes[1].imshow(fusion_weight.reshape(1, -1), cmap='RdBu_r', 
                        aspect='auto', vmin=-2, vmax=2)
    axes[1].set_xticks(range(len(fusion_weight)))
    axes[1].set_yticks([])
    axes[1].set_xlabel('Head Index')
    axes[1].set_title('Fusion Weights Heatmap')
    plt.colorbar(im, ax=axes[1])
    
    # 在热力图上添加数值
    for i, v in enumerate(fusion_weight):
        color = 'white' if abs(v) > 1 else 'black'
        axes[1].text(i, 0, f'{v:.2f}', ha='center', va='center', 
                    color=color, fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved fusion_weight visualization to {save_path}")
    plt.close()


def visualize_counter(state_dict: OrderedDict, save_path: str = "counter.png"):
    """可视化 counter 卷积核"""
    if 'counter.weight' not in state_dict:
        print("counter.weight not found")
        return
    
    counter = state_dict['counter.weight']
    if isinstance(counter, torch.Tensor):
        counter = counter.cpu().numpy()
    
    # 去除多余的维度
    while counter.ndim > 2:
        counter = counter.squeeze(0)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    im = ax.imshow(counter, cmap='RdBu_r', vmin=-0.2, vmax=0.2)
    ax.set_title('Counter Kernel (3x3)\nFixed Neighborhood Counter', fontsize=14)
    
    # 添加数值标签
    for i in range(counter.shape[0]):
        for j in range(counter.shape[1]):
            text = ax.text(j, i, f'{counter[i, j]:.4f}',
                          ha="center", va="center",
                          color="white" if abs(counter[i, j]) > 0.1 else "black",
                          fontsize=14, fontweight='bold')
    
    # 标记中心（当前细胞）
    center = counter.shape[0] // 2
    ax.add_patch(plt.Rectangle((center-0.5, center-0.5), 1, 1, 
                               fill=False, edgecolor='green', linewidth=3))
    ax.text(center, -0.8, 'Current Cell\n(not counted)', 
            ha='center', fontsize=10, color='green')
    
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, label='Weight Value')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved counter visualization to {save_path}")
    plt.close()


def visualize_head(state_dict: OrderedDict, head_idx: int, save_path: str = None):
    """可视化单个 head 的权重"""
    prefix = f'heads.{head_idx}'
    
    # 获取该 head 的所有参数
    w1_key = f'{prefix}.0.weight'
    b1_key = f'{prefix}.0.bias'
    w2_key = f'{prefix}.2.weight'
    b2_key = f'{prefix}.2.bias'
    
    if w1_key not in state_dict:
        print(f"Head {head_idx} not found")
        return
    
    w1 = state_dict[w1_key].cpu().numpy() if isinstance(state_dict[w1_key], torch.Tensor) else state_dict[w1_key]
    b1 = state_dict[b1_key].cpu().numpy() if isinstance(state_dict[b1_key], torch.Tensor) else state_dict[b1_key]
    w2 = state_dict[w2_key].cpu().numpy() if isinstance(state_dict[w2_key], torch.Tensor) else state_dict[w2_key]
    b2 = state_dict[b2_key].cpu().numpy() if isinstance(state_dict[b2_key], torch.Tensor) else state_dict[b2_key]
    
    # 调整形状
    # w1: [hidden_dim, 2, 1, 1] -> [hidden_dim, 2]
    w1 = w1.squeeze().reshape(-1, 2)
    # w2: [2, hidden_dim, 1, 1] -> [2, hidden_dim]
    w2 = w2.squeeze().reshape(2, -1)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Head {head_idx} Visualization', fontsize=16, fontweight='bold')
    
    # Layer 1 权重
    im1 = axes[0, 0].imshow(w1, aspect='auto', cmap='RdBu_r')
    axes[0, 0].set_xlabel('Input (0: Cell State, 1: Neighbor Count)')
    axes[0, 0].set_ylabel('Hidden Unit')
    axes[0, 0].set_title('Layer 1: Weights (input -> hidden)')
    axes[0, 0].set_xticks([0, 1])
    axes[0, 0].set_xticklabels(['Cell', 'Count'])
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Layer 1 偏置
    axes[0, 1].bar(range(len(b1)), b1, color='steelblue', edgecolor='black')
    axes[0, 1].set_xlabel('Hidden Unit')
    axes[0, 1].set_ylabel('Bias Value')
    axes[0, 1].set_title('Layer 1: Biases')
    axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Layer 2 权重
    im2 = axes[1, 0].imshow(w2, aspect='auto', cmap='RdBu_r')
    axes[1, 0].set_xlabel('Hidden Unit')
    axes[1, 0].set_ylabel('Output (0: Dead, 1: Alive)')
    axes[1, 0].set_title('Layer 2: Weights (hidden -> output)')
    axes[1, 0].set_yticks([0, 1])
    axes[1, 0].set_yticklabels(['Dead', 'Alive'])
    plt.colorbar(im2, ax=axes[1, 0])
    
    # Layer 2 偏置
    axes[1, 1].bar(['Dead', 'Alive'], b2, color=['coral', 'lightgreen'], 
                   edgecolor='black')
    axes[1, 1].set_ylabel('Bias Value')
    axes[1, 1].set_title('Layer 2: Output Biases')
    axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    # 在柱状图上添加数值
    for i, v in enumerate(b2):
        axes[1, 1].text(i, v, f'{v:.3f}', ha='center', 
                       va='bottom' if v > 0 else 'top', fontsize=10)
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = f"head_{head_idx}.png"
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved head {head_idx} visualization to {save_path}")
    plt.close()


def visualize_all_heads_summary(state_dict: OrderedDict, save_path: str = "all_heads_summary.png"):
    """可视化所有 heads 的汇总信息"""
    # 收集所有 heads 的信息
    head_indices = []
    for key in state_dict.keys():
        match = re.match(r'heads\.(\d+)\.0\.weight', key)
        if match:
            head_indices.append(int(match.group(1)))
    
    num_heads = len(head_indices)
    print(f"Found {num_heads} heads")
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, head_idx in enumerate(head_indices[:8]):  # 最多显示8个
        prefix = f'heads.{head_idx}'
        w1 = state_dict[f'{prefix}.0.weight']
        if isinstance(w1, torch.Tensor):
            w1 = w1.cpu().numpy()
        w1 = w1.squeeze().reshape(-1, 2)
        
        # 计算每个 head 的权重统计
        cell_weight_mean = np.mean(w1[:, 0])
        count_weight_mean = np.mean(w1[:, 1])
        
        # 绘制权重分布
        ax = axes[idx]
        x = np.arange(w1.shape[0])
        width = 0.35
        
        ax.bar(x - width/2, w1[:, 0], width, label='Cell State', alpha=0.8)
        ax.bar(x + width/2, w1[:, 1], width, label='Neighbor Count', alpha=0.8)
        ax.set_xlabel('Hidden Unit')
        ax.set_ylabel('Weight Value')
        ax.set_title(f'Head {head_idx}\nFusion Weight: {state_dict["fusion_weight"][head_idx]:.3f}')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('All Heads: Layer 1 Weight Distribution', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved all heads summary to {save_path}")
    plt.close()


def visualize_decision_heatmap(state_dict: OrderedDict, save_path: str = "decision_heatmap.png"):
    """
    可视化每个 head 对 [cell_state, neighbor_count] 组合的决策输出
    """
    # 收集所有 heads 的决策权重
    head_indices = []
    for key in state_dict.keys():
        match = re.match(r'heads\.(\d+)\.0\.weight', key)
        if match:
            head_indices.append(int(match.group(1)))
    
    num_heads = len(head_indices)
    
    # 计算每个 head 对每种 [cell_state, neighbor_count] 组合的输出
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()
    
    for idx, head_idx in enumerate(head_indices[:8]):
        prefix = f'heads.{head_idx}'
        
        w1 = state_dict[f'{prefix}.0.weight']
        b1 = state_dict[f'{prefix}.0.bias']
        w2 = state_dict[f'{prefix}.2.weight']
        b2 = state_dict[f'{prefix}.2.bias']
        
        if isinstance(w1, torch.Tensor):
            w1 = w1.cpu().numpy()
            b1 = b1.cpu().numpy()
            w2 = w2.cpu().numpy()
            b2 = b2.cpu().numpy()
        
        w1 = w1.squeeze().reshape(-1, 2)
        w2 = w2.squeeze().reshape(2, -1)
        
        # 计算所有 [cell_state, neighbor_count] 组合的输出
        # cell_state: 0 或 1
        # neighbor_count: 0-8
        heatmap = np.zeros((2, 9))  # [cell_state, neighbor_count]
        
        for cell_state in [0, 1]:
            for neighbor_count in range(9):
                x = np.array([cell_state, neighbor_count])
                # Layer 1
                h = np.maximum(0, w1 @ x + b1)  # ReLU
                # Layer 2
                logits = w2 @ h + b2
                # Alive 概率
                prob = 1 / (1 + np.exp(-(logits[1] - logits[0])))
                heatmap[cell_state, neighbor_count] = prob
        
        ax = axes[idx]
        im = ax.imshow(heatmap, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        ax.set_xlabel('Number of Neighbors')
        ax.set_ylabel('Current Cell State')
        ax.set_title(f'Head {head_idx}\nP(Alive)')
        ax.set_xticks(range(9))
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Dead (0)', 'Alive (1)'])
        
        # 添加数值标签
        for i in range(2):
            for j in range(9):
                text = ax.text(j, i, f'{heatmap[i, j]:.2f}',
                              ha="center", va="center",
                              color="white" if heatmap[i, j] < 0.5 else "black",
                              fontsize=10, fontweight='bold')
        
        plt.colorbar(im, ax=ax)
    
    plt.suptitle('Decision Heatmaps: P(Alive | Cell State, Neighbor Count)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved decision heatmap to {save_path}")
    plt.close()


def extract_ensemble_rule(state_dict: OrderedDict) -> np.ndarray:
    """
    提取集成规则：加权平均所有 heads 的决策
    """
    fusion_weight = state_dict['fusion_weight']
    if isinstance(fusion_weight, torch.Tensor):
        fusion_weight = fusion_weight.cpu().numpy()
    
    # 归一化融合权重
    fusion_weight = np.exp(fusion_weight) / np.sum(np.exp(fusion_weight))
    
    head_indices = []
    for key in state_dict.keys():
        match = re.match(r'heads\.(\d+)\.0\.weight', key)
        if match:
            head_indices.append(int(match.group(1)))
    
    # 集成决策
    ensemble_heatmap = np.zeros((2, 9))
    
    for head_idx in head_indices:
        prefix = f'heads.{head_idx}'
        
        w1 = state_dict[f'{prefix}.0.weight']
        b1 = state_dict[f'{prefix}.0.bias']
        w2 = state_dict[f'{prefix}.2.weight']
        b2 = state_dict[f'{prefix}.2.bias']
        
        if isinstance(w1, torch.Tensor):
            w1 = w1.cpu().numpy()
            b1 = b1.cpu().numpy()
            w2 = w2.cpu().numpy()
            b2 = b2.cpu().numpy()
        
        w1 = w1.squeeze().reshape(-1, 2)
        w2 = w2.squeeze().reshape(2, -1)
        
        for cell_state in [0, 1]:
            for neighbor_count in range(9):
                x = np.array([cell_state, neighbor_count])
                h = np.maximum(0, w1 @ x + b1)
                logits = w2 @ h + b2
                prob = 1 / (1 + np.exp(-(logits[1] - logits[0])))
                ensemble_heatmap[cell_state, neighbor_count] += fusion_weight[head_idx] * prob
    
    return ensemble_heatmap


def visualize_ensemble_rule(state_dict: OrderedDict, save_path: str = "ensemble_rule.png"):
    """可视化集成后的规则"""
    fusion_weight = state_dict['fusion_weight']
    if isinstance(fusion_weight, torch.Tensor):
        fusion_weight = fusion_weight.cpu().numpy()
    
    ensemble_heatmap = extract_ensemble_rule(state_dict)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 融合权重
    ax = axes[0]
    colors = plt.cm.RdYlGn((fusion_weight - fusion_weight.min()) / 
                           (fusion_weight.max() - fusion_weight.min() + 1e-8))
    bars = ax.bar(range(len(fusion_weight)), fusion_weight, color=colors, edgecolor='black')
    ax.set_xlabel('Head Index')
    ax.set_ylabel('Fusion Weight (raw)')
    ax.set_title('Raw Fusion Weights')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    for i, (bar, v) in enumerate(zip(bars, fusion_weight)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{v:.2f}', ha='center', va='bottom' if height > 0 else 'top',
                fontsize=9)
    
    # Softmax 后的权重
    ax = axes[1]
    softmax_weight = np.exp(fusion_weight) / np.sum(np.exp(fusion_weight))
    bars = ax.bar(range(len(softmax_weight)), softmax_weight, 
                  color='steelblue', edgecolor='black')
    ax.set_xlabel('Head Index')
    ax.set_ylabel('Fusion Weight (softmax)')
    ax.set_title('Softmax-normalized Fusion Weights')
    ax.grid(axis='y', alpha=0.3)
    
    for i, (bar, v) in enumerate(zip(bars, softmax_weight)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{v:.3f}', ha='center', va='bottom',
                fontsize=9)
    
    # 集成决策热力图
    ax = axes[2]
    im = ax.imshow(ensemble_heatmap, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax.set_xlabel('Number of Neighbors')
    ax.set_ylabel('Current Cell State')
    ax.set_title('Ensemble Decision: P(Alive)')
    ax.set_xticks(range(9))
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Dead (0)', 'Alive (1)'])
    
    for i in range(2):
        for j in range(9):
            text = ax.text(j, i, f'{ensemble_heatmap[i, j]:.2f}',
                          ha="center", va="center",
                          color="white" if ensemble_heatmap[i, j] < 0.5 else "black",
                          fontsize=12, fontweight='bold')
    
    plt.colorbar(im, ax=ax)
    
    # 提取 B/S 规则
    birth = []
    survival = []
    for neighbor_count in range(9):
        if ensemble_heatmap[0, neighbor_count] > 0.5:
            birth.append(neighbor_count)
        if ensemble_heatmap[1, neighbor_count] > 0.5:
            survival.append(neighbor_count)
    
    fig.suptitle(f'Extracted Rule: B{",".join(map(str, birth))}/S{",".join(map(str, survival))}', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved ensemble rule to {save_path}")
    plt.close()
    
    return ensemble_heatmap


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize state_dict parameters")
    parser.add_argument("--input", type=str, required=True,
                       help="Path to state_dict file (.pth or .txt)")
    parser.add_argument("--output_dir", type=str, default="./visualization_output",
                       help="Output directory for visualizations")
    parser.add_argument("--heads", type=str, default="all",
                       help="Which heads to visualize (e.g., '0,1,2' or 'all')")
    args = parser.parse_args()
    
    # 创建输出目录
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载 state_dict
    print(f"Loading state_dict from {args.input}")
    state_dict = load_state_dict(args.input)
    print(f"Loaded {len(state_dict)} parameters")
    
    # 列出所有参数
    print("\nParameters:")
    for key in state_dict.keys():
        shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'N/A'
        print(f"  {key}: {shape}")
    
    # 可视化 fusion_weight
    print("\nVisualizing fusion_weight...")
    visualize_fusion_weight(state_dict, 
                           os.path.join(args.output_dir, "fusion_weight.png"))
    
    # 可视化 counter
    print("\nVisualizing counter...")
    visualize_counter(state_dict, 
                     os.path.join(args.output_dir, "counter.png"))
    
    # 可视化单个 heads
    if args.heads == "all":
        head_indices = []
        for key in state_dict.keys():
            match = re.match(r'heads\.(\d+)\.0\.weight', key)
            if match:
                head_indices.append(int(match.group(1)))
    else:
        head_indices = [int(x) for x in args.heads.split(",")]
    
    print(f"\nVisualizing heads: {head_indices}")
    for head_idx in head_indices:
        visualize_head(state_dict, head_idx,
                      os.path.join(args.output_dir, f"head_{head_idx}.png"))
    
    # 可视化所有 heads 汇总
    print("\nVisualizing all heads summary...")
    visualize_all_heads_summary(state_dict,
                               os.path.join(args.output_dir, "all_heads_summary.png"))
    
    # 可视化决策热力图
    print("\nVisualizing decision heatmaps...")
    visualize_decision_heatmap(state_dict,
                              os.path.join(args.output_dir, "decision_heatmap.png"))
    
    # 可视化集成规则
    print("\nVisualizing ensemble rule...")
    visualize_ensemble_rule(state_dict,
                           os.path.join(args.output_dir, "ensemble_rule.png"))
    
    print(f"\nAll visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()
