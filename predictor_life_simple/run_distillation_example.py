"""
知识蒸馏使用示例

演示如何：
1. 加载预训练的教师网络
2. 创建并初始化学生网络
3. 执行蒸馏训练
4. 提取并可视化规则

Usage:
    python run_distillation_example.py --teacher_checkpoint path/to/teacher.pth
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from model_conv import SimpleCNNSmall, SimpleCNNSmall2Layer
from distillation import InterpretableCA, extract_and_print_rule


def visualize_counter_kernel(model: InterpretableCA, save_path: str = "counter_kernel.png"):
    """可视化学生网络的计数器卷积核"""
    kernel = model.get_counter_kernel().squeeze().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(kernel, cmap='RdYlBu_r', vmin=-0.5, vmax=0.5)
    ax.set_title("Student Network: Counter Kernel (Fixed)")
    
    # 在格子上标注数值
    for i in range(3):
        for j in range(3):
            text = ax.text(j, i, f'{kernel[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=12)
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved counter kernel visualization to {save_path}")
    plt.close()


def visualize_decision_weights(model: InterpretableCA, save_path: str = "decision_weights.png"):
    """可视化决策层的权重"""
    # 获取第一层决策权重 [hidden_dim, 2, 1, 1] -> [hidden_dim, 2]
    w1 = model.decision[0].weight.squeeze().cpu().detach().numpy()  # [hidden_dim, 2]
    b1 = model.decision[0].bias.squeeze().cpu().detach().numpy() if model.decision[0].bias is not None else np.zeros(w1.shape[0])
    
    # 获取第二层决策权重 [2, hidden_dim, 1, 1] -> [2, hidden_dim]
    w2 = model.decision[2].weight.squeeze().cpu().detach().numpy()  # [2, hidden_dim]
    b2 = model.decision[2].bias.squeeze().cpu().detach().numpy() if model.decision[2].bias is not None else np.zeros(2)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 第一层权重热力图
    im1 = axes[0, 0].imshow(w1, aspect='auto', cmap='RdBu_r')
    axes[0, 0].set_xlabel('Input Feature (Cell State, Neighbor Count)')
    axes[0, 0].set_ylabel('Hidden Unit')
    axes[0, 0].set_title('Decision Layer 1: Weights')
    axes[0, 0].set_xticks([0, 1])
    axes[0, 0].set_xticklabels(['Cell State', 'Neighbor Count'])
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 第一层偏置
    axes[0, 1].bar(range(len(b1)), b1)
    axes[0, 1].set_xlabel('Hidden Unit')
    axes[0, 1].set_ylabel('Bias')
    axes[0, 1].set_title('Decision Layer 1: Biases')
    
    # 第二层权重热力图
    im2 = axes[1, 0].imshow(w2, aspect='auto', cmap='RdBu_r')
    axes[1, 0].set_xlabel('Hidden Unit')
    axes[1, 0].set_ylabel('Output Class (Dead, Alive)')
    axes[1, 0].set_title('Decision Layer 2: Weights')
    axes[1, 0].set_yticks([0, 1])
    axes[1, 0].set_yticklabels(['Dead', 'Alive'])
    plt.colorbar(im2, ax=axes[1, 0])
    
    # 第二层偏置
    axes[1, 1].bar(['Dead', 'Alive'], b2)
    axes[1, 1].set_ylabel('Bias')
    axes[1, 1].set_title('Decision Layer 2: Biases')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved decision weights visualization to {save_path}")
    plt.close()


def visualize_rule_heatmap(rule_table: dict, save_path: str = "rule_heatmap.png"):
    """
    将规则表可视化为热力图
    
    显示不同 [细胞状态, 邻居计数] 组合下的预测结果
    """
    # 构建热力图数据
    heatmap_data = np.zeros((2, 9))  # [cell_state, neighbor_count]
    prob_data = np.zeros((2, 9))
    
    for (cell_state, neighbor_count), info in rule_table.items():
        heatmap_data[cell_state, neighbor_count] = info['prediction']
        prob_data[cell_state, neighbor_count] = info['probability']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 预测结果热力图
    im1 = axes[0].imshow(heatmap_data, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    axes[0].set_xlabel('Number of Neighbors')
    axes[0].set_ylabel('Current Cell State')
    axes[0].set_title('Predicted Next State\n(Green=Alive, Red=Dead)')
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(['Dead (0)', 'Alive (1)'])
    axes[0].set_xticks(range(9))
    
    # 在格子上标注
    for i in range(2):
        for j in range(9):
            text = axes[0].text(j, i, f'{int(heatmap_data[i, j])}',
                               ha="center", va="center", 
                               color="white" if heatmap_data[i, j] > 0.5 else "black",
                               fontsize=14, fontweight='bold')
    
    plt.colorbar(im1, ax=axes[0], ticks=[0, 1])
    
    # 预测置信度热力图
    im2 = axes[1].imshow(prob_data, cmap='YlOrRd', vmin=0.5, vmax=1.0, aspect='auto')
    axes[1].set_xlabel('Number of Neighbors')
    axes[1].set_ylabel('Current Cell State')
    axes[1].set_title('Prediction Confidence')
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(['Dead (0)', 'Alive (1)'])
    axes[1].set_xticks(range(9))
    
    # 在格子上标注置信度
    for i in range(2):
        for j in range(9):
            text = axes[1].text(j, i, f'{prob_data[i, j]:.2f}',
                               ha="center", va="center", 
                               color="white" if prob_data[i, j] > 0.75 else "black",
                               fontsize=11)
    
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved rule heatmap to {save_path}")
    plt.close()


def compare_teacher_student(teacher_model: nn.Module, student_model: InterpretableCA, 
                            test_input: torch.Tensor, save_path: str = "comparison.png"):
    """
    对比教师网络和学生网络的输出
    """
    teacher_model.eval()
    student_model.eval()
    
    with torch.no_grad():
        teacher_out = teacher_model(test_input)
        if isinstance(teacher_out, tuple):
            teacher_out = teacher_out[0]
        
        student_out = student_model(test_input)
        
        # 转换为概率
        teacher_prob = torch.softmax(teacher_out, dim=1)
        student_prob = torch.softmax(student_out, dim=1)
    
    # 可视化第一个样本
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 输入
    axes[0, 0].imshow(test_input[0, 0].cpu().numpy(), cmap='gray')
    axes[0, 0].set_title('Input State')
    axes[0, 0].axis('off')
    
    # 教师网络输出（Alive通道概率）
    im1 = axes[0, 1].imshow(teacher_prob[0, 1].cpu().numpy(), cmap='hot', vmin=0, vmax=1)
    axes[0, 1].set_title('Teacher: P(Alive)')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # 学生网络输出（Alive通道概率）
    im2 = axes[0, 2].imshow(student_prob[0, 1].cpu().numpy(), cmap='hot', vmin=0, vmax=1)
    axes[0, 2].set_title('Student: P(Alive)')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2])
    
    # 教师网络预测
    axes[1, 0].imshow(teacher_prob[0].argmax(0).cpu().numpy(), cmap='gray')
    axes[1, 0].set_title('Teacher: Prediction')
    axes[1, 0].axis('off')
    
    # 学生网络预测
    axes[1, 1].imshow(student_prob[0].argmax(0).cpu().numpy(), cmap='gray')
    axes[1, 1].set_title('Student: Prediction')
    axes[1, 1].axis('off')
    
    # 差异图
    diff = (teacher_prob[0, 1] - student_prob[0, 1]).abs().cpu().numpy()
    im3 = axes[1, 2].imshow(diff, cmap='Reds', vmin=0, vmax=diff.max())
    axes[1, 2].set_title(f'Absolute Difference\nMean: {diff.mean():.4f}')
    axes[1, 2].axis('off')
    plt.colorbar(im3, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved teacher-student comparison to {save_path}")
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run distillation example and visualization")
    parser.add_argument("--teacher_checkpoint", type=str, required=True,
                       help="Path to teacher model checkpoint")
    parser.add_argument("--student_checkpoint", type=str, default=None,
                       help="Path to student model checkpoint (optional)")
    parser.add_argument("--teacher_model", type=str, default="SimpleCNNSmall",
                       choices=["SimpleCNNSmall", "SimpleCNNSmall2Layer"])
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载教师网络
    print("\n" + "="*60)
    print("Loading Teacher Network")
    print("="*60)
    teacher_class = globals()[args.teacher_model]
    teacher_model = teacher_class().to(device)
    teacher_model.load_state_dict(torch.load(args.teacher_checkpoint, map_location=device))
    teacher_model.eval()
    print(f"Loaded {args.teacher_model} from {args.teacher_checkpoint}")
    
    # 创建/加载学生网络
    print("\n" + "="*60)
    print("Creating Student Network")
    print("="*60)
    student_model = InterpretableCA(hidden_dim=16).to(device)
    
    if args.student_checkpoint:
        student_model.load_state_dict(torch.load(args.student_checkpoint, map_location=device))
        print(f"Loaded student from {args.student_checkpoint}")
    else:
        print("Using untrained student (random initialization)")
    
    student_model.eval()
    
    # 可视化计数器卷积核
    print("\n" + "="*60)
    print("Visualizing Counter Kernel")
    print("="*60)
    visualize_counter_kernel(student_model)
    
    # 提取规则
    print("\n" + "="*60)
    print("Extracting Rule from Student")
    print("="*60)
    rule_table = extract_and_print_rule(student_model)
    
    # 可视化规则热力图
    print("\n" + "="*60)
    print("Visualizing Rule Heatmap")
    print("="*60)
    visualize_rule_heatmap(rule_table)
    
    # 可视化决策层权重
    print("\n" + "="*60)
    print("Visualizing Decision Layer Weights")
    print("="*60)
    visualize_decision_weights(student_model)
    
    # 对比教师和学生网络
    print("\n" + "="*60)
    print("Comparing Teacher and Student")
    print("="*60)
    test_input = torch.randint(0, 2, (1, 1, 50, 50), dtype=torch.float32).to(device)
    compare_teacher_student(teacher_model, student_model, test_input)
    
    print("\n" + "="*60)
    print("All visualizations saved!")
    print("="*60)


if __name__ == "__main__":
    main()
