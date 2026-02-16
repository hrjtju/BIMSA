"""
知识蒸馏模块：将教师网络（黑盒CNN）的知识蒸馏到学生网络（可解释CA网络）

学生网络结构：
- 第一层：固定卷积核（邻居计数器），权重为归一化的1，不可训练
- 第二层：全连接决策层，实现 "计数 + 查表" 的逻辑

参考：与Kimi讨论的两阶段发现框架
"""

import argparse
import toml
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.nn.functional import softmax, cross_entropy, mse_loss, kl_div, log_softmax
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from einops import rearrange
from jaxtyping import Float, Array

from dataloader import get_dataloader
from args import Args
import model_conv

import os
import re

BIMSA_LIFE_DIR = os.environ.get('BIMSA_LIFE_DIR', "./predictor_life_simple/datasets")


class InterpretableCA(nn.Module):
    """
    可解释的元胞自动机学生网络
    
    结构对应于CA的数学分解：f_cell(c, n) = g(c, sum(n_i))
    - counter: 固定为邻居计数器（Moore邻域，8个邻居）
    - decision: 全连接决策层，学习"计数+查表"逻辑
    """
    __version__ = '0.1.0-distill'

    def __init__(self, hidden_dim: int = 64, num_classes: int = 2, num_heads: int = 8):
        """
        Args:
            hidden_dim: 决策层隐藏层维度
            num_classes: 输出类别数（二值CA为2）
        """
        super().__init__()
        
        # 第一层：固定邻居计数器（不可训练）
        # 使用 3x3 卷积核，中心为0（不计入自身），周围8格为1（邻居）
        self.counter = nn.Conv2d(1, 2, kernel_size=3, padding=1, bias=False, padding_mode="circular")
        # 初始化计数核：Moore邻域
        count_kernel = torch.tensor([[[1., 1., 1.],
                                    [1., 0., 1.],
                                    [1., 1., 1.]]]).view(1, 1, 3, 3)
        # 归一化（可选，但有助于数值稳定性）
        count_kernel = count_kernel / count_kernel.sum()
        self.counter.weight.data = count_kernel
        self.counter.weight.requires_grad = False  # 固定不可训练
        
        # 决策层：输入为 [细胞自身状态, 邻居计数] -> 输出下一状态
        # 使用1x1卷积实现全连接（保持空间结构）
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2, hidden_dim, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, num_classes, kernel_size=1)
            ) for _ in range(num_heads)
        ])

        # TODO: Routing
        self.fusion_weight = nn.Parameter(torch.ones(num_heads))

    def forward(self, x: Float[Array, "batch 1 h w"]) -> Float[Array, "batch 2 h w"]:
        """
        Args:
            x: 输入网格，形状 [batch, 1, H, W]
        Returns:
            输出logits，形状 [batch, 2, H, W]
        """
        # 计算邻居计数
        neighbor_count = self.counter(x)  # [batch, 1, H, W]
        
        # 拼接特征：[细胞状态, 邻居计数]
        features = torch.cat([x, neighbor_count], dim=1)  # [batch, 2, H, W]
        
        # 决策层
        head_outputs = [head(features) for head in self.heads]
        output = torch.stack(head_outputs, dim=0)  # [num_heads, B, 2, H, W]
        
        weights = softmax(self.fusion_weight, dim=0)
        output = torch.einsum('n,nbchw->bchw', weights, output)
        
        return output
    
    def get_counter_kernel(self) -> Tensor:
        """获取计数器卷积核（用于可视化）"""
        return self.counter.weight.data.clone()
    
    def extract_rule_table(self, threshold: float = 0.5) -> dict:
        """
        从决策层提取规则表
        
        遍历所有可能的 [细胞状态, 邻居计数] 组合，记录决策输出
        
        Returns:
            dict: 键为 (cell_state, neighbor_count)，值为预测类别
        """
        self.eval()
        rule_table = {}
        
        with torch.no_grad():
            for s in [0, 1]:
                for k in range(9):
                    inp = torch.tensor([[s, k/8]], dtype=torch.float32)
                    inp = inp.view(1, 2, 1, 1).to(next(self.parameters()).device)
                    
                    # 完整前向（含所有头融合）
                    logits = self.forward(inp[:, 0:1, :, :])  # 注意：forward期望[B,1,H,W]
                    # 上面构造的inp已经是[B,2,1,1]，但forward内部会重新算count
                    # 应该直接调用各组件：
                    features = inp
                    head_outputs = [head(features) for head in self.heads]
                    weights = softmax(self.fusion_weight, dim=0)
                    output = sum(w * h for w, h in zip(weights, head_outputs))
                    
                    probs = softmax(output.view(-1), dim=0)
                    pred = output.argmax(dim=1).item()
                    
                    rule_table[(s, k)] = {
                        'prediction': pred,
                        'probability': probs[pred].item(),
                        'logits': output.view(-1).tolist()
                    }
        
        return rule_table

def distillation_loss(
    student: InterpretableCA, 
    student_logits: Tensor,
    teacher_logits: Tensor,
    labels: Tensor,
    temperature: float = 4.0,
    alpha: float = 0.5,
) -> Tensor:
    """
    知识蒸馏损失函数（Hinton et al., 2015）
    
    L = alpha * L_CE(student, true_labels) + (1-alpha) * T^2 * KL(student/T, teacher/T)
    
    Args:
        student_logits: 学生网络输出 [batch, C, H, W]
        teacher_logits: 教师网络输出 [batch, C, H, W]
        labels: 真实标签 [batch, H, W]
        temperature: 蒸馏温度
        alpha: 硬标签损失权重（1-alpha为软标签损失权重）
    
    Returns:
        总损失
    """
    batch_size, num_classes, h, w = student_logits.shape
    
    # 硬标签损失：交叉熵
    student_softmax = softmax(rearrange(student_logits, "b c h w -> (b h w) c"), dim=-1)
    labels_flat = rearrange(labels, "b h w -> (b h w)").long()
    weight = torch.tensor([1.0, 10.0]).cuda()
    hard_loss = cross_entropy(student_softmax, labels_flat, weight)
    
    # 软标签损失：KL散度
    # 对logits进行温度缩放
    student_soft = log_softmax(rearrange(student_logits / temperature, "b c h w -> (b h w) c"), dim=-1)
    teacher_soft = softmax(rearrange(teacher_logits / temperature, "b c h w -> (b h w) c"), dim=-1)
    soft_loss = kl_div(student_soft, teacher_soft, reduction='batchmean') * (temperature ** 2)
    
    l1_reg = 0
    param_all = 0
    for name, param in student.named_parameters():
        if 'weight' in name and param.requires_grad == True:
            l1_reg = l1_reg + torch.linalg.vector_norm(param, ord=1, dim=None)
            param_all += param.numel()
    
    # 总损失
    total_loss = alpha * hard_loss + (1 - alpha) * soft_loss + 0.1 * l1_reg / param_all
    
    return total_loss, hard_loss, soft_loss, l1_reg


def train_student(
    teacher_model: nn.Module,
    student_model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    args: dict,
    device: torch.device,
) -> nn.Module:
    """
    训练学生网络（知识蒸馏）
    
    Args:
        teacher_model: 已训练好的教师网络
        student_model: 待训练的学生网络
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        args: 配置参数字典
        device: 计算设备
    
    Returns:
        训练好的学生网络
    """
    teacher_model.eval()
    student_model.to(device)
    teacher_model.to(device)
    
    # 优化器（只优化学生网络的可训练参数）
    optimizer = getattr(optim, args["optimizer"]["name"])(
        filter(lambda p: p.requires_grad, student_model.parameters()),
        **args["optimizer"]["args"]
    )
    
    # 学习率调度器
    use_lr_scheduler = args["lr_scheduler"].get("name", None) is not None
    if use_lr_scheduler:
        scheduler = getattr(optim.lr_scheduler, args["lr_scheduler"]["name"])(
            optimizer, **args["lr_scheduler"]["args"]
        )
    
    temperature = args["distillation"]["temperature"]
    alpha = args["distillation"]["alpha"]
    epochs = args["training"]["epochs"]
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        print("-" * 10)
        
        # 训练阶段
        student_model.train()
        running_loss = 0.0
        running_hard_loss = 0.0
        running_soft_loss = 0.0
        running_reg_loss = 0.0
        correct = 0
        total = 0
        
        with tqdm(train_loader) as p:
            for idx, (inputs, labels) in enumerate(p):
                inputs, labels = inputs.to(device), labels.to(device)
                
                # 获取教师网络输出（无梯度）
                with torch.no_grad():
                    teacher_outputs = teacher_model(inputs)
                    # 教师网络输出可能是tuple（如果包含reconstruction）
                    if isinstance(teacher_outputs, tuple):
                        teacher_outputs = teacher_outputs[0]
                
                # 学生网络前向
                student_outputs = student_model(inputs)
                
                # 计算蒸馏损失
                loss, hard_loss, soft_loss, l1_reg = distillation_loss(
                    student_model, 
                    student_outputs, teacher_outputs, labels,
                    temperature=temperature, alpha=alpha
                )
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                norm = torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, student_model.parameters()),
                    max_norm=100.0
                )
                
                optimizer.step()
                if use_lr_scheduler:
                    scheduler.step()
                
                # 统计
                running_loss += loss.item()
                running_hard_loss += hard_loss.item()
                running_soft_loss += soft_loss.item()
                running_reg_loss += l1_reg.item()
                
                predicted = student_outputs.argmax(1)
                total += labels.numel()
                correct += predicted.eq(labels).sum().item()
                
                # 记录到wandb
                if idx % 10 == 0:
                    wandb.log(
                        data={
                        "train/total_loss": loss.item(),
                        "train/hard_loss": hard_loss.item(),
                        "train/soft_loss": soft_loss.item(),
                        "train/reg_loss": l1_reg.item(),
                        "train/gradient_norm": norm.item(),
                        },
                        step=idx
                        )
                
                p.set_postfix_str(f"loss {loss.item():.4e}")
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%")
        # wandb.log({
        #     "distill/train_epoch_loss": epoch_loss,
        #     "distill/train_epoch_acc": epoch_acc,
        #     "distill/train_hard_loss": running_hard_loss / len(train_loader),
        #     "distill/train_soft_loss": running_soft_loss / len(train_loader),
        # })
        
        # 验证阶段
        student_model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = student_model(inputs)
                predicted = outputs.argmax(1)
                
                val_total += labels.numel()
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        print(f"Val Acc: {val_acc:.2f}%")
        wandb.log({"val/val_epoch_acc": val_acc})
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(student_model.state_dict(), 
                      f'best_student_{student_model.__version__}.pth')
            print(f"Saved best model with acc: {best_acc:.2f}%")
        
        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            torch.save(student_model.state_dict(),
                      f'checkpoint_student_epoch{epoch+1}.pth')
        
        if epoch_loss < 0.05 or val_acc > 99:
            break
    
    return student_model


def extract_and_print_rule(student_model: InterpretableCA):
    """
    从训练好的学生网络中提取并打印规则表
    """
    print("\n" + "="*60)
    print("Extracted Rule Table from Student Network")
    print("="*60)
    
    rule_table = student_model.extract_rule_table()
    
    # 整理为B/S格式
    birth_conditions = []
    survival_conditions = []
    
    for cell_state in [0, 1]:
        for neighbor_count in range(9):
            info = rule_table[(cell_state, neighbor_count)]
            pred = info['prediction']
            prob = info['probability']
            
            if cell_state == 0 and pred == 1:  # 死亡 -> 存活（Birth）
                birth_conditions.append((neighbor_count, prob))
            elif cell_state == 1 and pred == 1:  # 存活 -> 存活（Survival）
                survival_conditions.append((neighbor_count, prob))
    
    birth_str = "".join([str(n) for n, _ in sorted(birth_conditions)])
    survival_str = "".join([str(n) for n, _ in sorted(survival_conditions)])
    
    print(f"\nB{birth_str}/S{survival_str}")
    print(f"\nBirth conditions: {birth_conditions}")
    print(f"Survival conditions: {survival_conditions}")
    
    # 打印完整表格
    print("\nFull Rule Table:")
    print("-" * 60)
    print(f"{'Cell State':<12} {'Neighbors':<10} {'Prediction':<12} {'Prob':<8}")
    print("-" * 60)
    for cell_state in [0, 1]:
        for neighbor_count in range(9):
            info = rule_table[(cell_state, neighbor_count)]
            print(f"{cell_state:<12} {neighbor_count:<10} {info['prediction']:<12} {info['probability']:.3f}")
    
    print("="*60)
    
    return rule_table


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Distill knowledge from teacher to interpretable student network")
    parser.add_argument("-p", "--hyperparameters", type=str, 
                       default="./predictor_life_simple/hyperparams/distill.toml",
                       help="Path to hyperparameters file")
    parser.add_argument("--teacher_checkpoint", type=str, required=True,
                       help="Path to trained teacher model checkpoint")
    parser.add_argument("--teacher_model", type=str, default="SimpleCNNSmall",
                       help="Teacher model architecture")
    # The above code seems to be a comment in Python. Comments in Python start with a hash symbol (#)
    # and are used to provide explanations or notes within the code. In this case, the comment is
    # indicating that the code is related to "args".
    args = parser.parse_args()
    
    # 加载配置
    args_dict = toml.load(args.hyperparameters)
    
    # 初始化wandb
    if args_dict["wandb"]["turn_on"]:
        wandb.init(project="predictor_life_distill", name=args_dict["wandb"]["entity"])
    else:
        wandb.init(mode="disabled")
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载教师网络
    teacher_class = getattr(model_conv, args.teacher_model)
    teacher_model = teacher_class().to(device)
    teacher_model.load_state_dict(torch.load(args.teacher_checkpoint, map_location=device))
    teacher_model.eval()
    print(f"Loaded teacher model from {args.teacher_checkpoint}")
    
    # 创建学生网络
    student_model = InterpretableCA(
        hidden_dim=args_dict["model"].get("hidden_dim", 32),
        num_classes=2
    )
    print(f"Created student model: {student_model.__version__}")
    print(f"Counter kernel:\n{student_model.get_counter_kernel().squeeze()}")
    
    # 匹配规则字符串
    s = re.findall(r"B\d*_S\d*", args.teacher_checkpoint)[0]
    
    data_dir = os.path.join(BIMSA_LIFE_DIR, f"200-200-{s}")
    assert os.path.exists(data_dir), f"data dir [{data_dir}] don't exist."
    print(f"Targeted data rule: {s}")
    
    # 数据加载器
    train_loader = get_dataloader(
        data_dir=data_dir,
        batch_size=args_dict["dataloader"]["train_batch_size"],
        shuffle=args_dict["dataloader"]["train_shuffle"],
        num_workers=args_dict["dataloader"]["train_num_workers"],
        split='train'
    )
    
    test_loader = get_dataloader(
        data_dir=data_dir,
        batch_size=args_dict["dataloader"]["test_batch_size"],
        shuffle=args_dict["dataloader"]["test_shuffle"],
        num_workers=args_dict["dataloader"]["test_num_workers"],
        split='test'
    )
    
    # 训练学生网络
    print("\nStarting distillation...")
    student_model = train_student(
        teacher_model=teacher_model,
        student_model=student_model,
        train_loader=train_loader,
        test_loader=test_loader,
        args=args_dict,
        device=device,
    )
    
    # 提取规则
    rule_table = extract_and_print_rule(student_model)
    
    # 保存最终模型
    torch.save(student_model.state_dict(), f'final_student_{student_model.__version__}.pth')
    print(f"\nSaved final student model")
    
    # print(student_model.state_dict())
    
    # 保存规则表
    import json
    with open('extracted_rule.json', 'w') as f:
        json.dump({f"{k[0]}_{k[1]}": v for k, v in rule_table.items()}, f, indent=2)
    print("Saved extracted rule to extracted_rule.json")
    
    with open('distilled.txt', 'w') as f:
        f.write(str(student_model.state_dict()))
    print("Saved student_model params to distilled.txt")


if __name__ == "__main__":
    main()
