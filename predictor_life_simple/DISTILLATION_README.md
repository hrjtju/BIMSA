# 知识蒸馏模块使用说明

## 概述

本模块实现了**可解释性知识蒸馏（Interpretable Knowledge Distillation）**，将黑盒教师网络（CNN）的知识迁移到结构受限的学生网络中，从而实现：

1. **结构发现**：通过教师网络卷积核分析确定Moore邻域
2. **白盒建模**：学生网络首层固定为邻居计数器，强制学习"计数+查表"逻辑
3. **规则提取**：从学生网络决策层直接读取B/S规则

## 核心思想

```
观测数据 → 黑盒学习（教师）→ 知识蒸馏 → 白盒模型（学生）→ 显式规则提取
```

学生网络结构对应CA的数学分解：

```
f_cell(c, n) = g(c, Σn_i)
```

- **counter**: 固定3×3卷积核，中心为0，周围8格为1，实现邻居计数
- **decision**: 可训练的全连接层，学习决策逻辑

## 文件说明

| 文件 | 说明 |
|------|------|
| `distillation.py` | 核心蒸馏模块，包含学生网络定义、蒸馏损失、训练流程 |
| `hyperparams/distill.toml` | 蒸馏训练配置文件 |
| `run_distillation_example.py` | 使用示例和可视化工具 |

## 使用方法

### 1. 准备教师网络

确保你有一个已训练好的教师网络检查点（如 `best_life_UNet_X.X.X.pth`）

### 2. 运行蒸馏训练

```bash
cd predictor_life_simple

python distillation.py \
    --teacher_checkpoint path/to/teacher.pth \
    --teacher_model SimpleCNNSmall \
    -p hyperparams/distill.toml
```

参数说明：
- `--teacher_checkpoint`: 教师网络检查点路径
- `--teacher_model`: 教师网络架构（SimpleCNNSmall / SimpleCNNSmall2Layer / SimpleCNNTiny）
- `-p`: 超参数配置文件路径

### 3. 可视化分析

```bash
python run_distillation_example.py \
    --teacher_checkpoint path/to/teacher.pth \
    --student_checkpoint path/to/student.pth \
    --teacher_model SimpleCNNSmall
```

这会生成以下可视化：
- `counter_kernel.png`: 学生网络的固定计数器卷积核
- `decision_weights.png`: 决策层权重可视化
- `rule_heatmap.png`: 提取的规则热力图
- `comparison.png`: 教师vs学生网络输出对比

## 配置文件说明 (`distill.toml`)

```toml
[model]
hidden_dim = 16          # 决策层隐藏层维度

[distillation]
temperature = 4.0        # 蒸馏温度（软标签平滑度）
alpha = 0.5              # 硬标签权重（1-alpha为软标签权重）

[training]
epochs = 50              # 训练轮数

[optimizer]
name = "Adam"
args = { lr = 0.001, weight_decay = 1e-4 }

[lr_scheduler]
name = "StepLR"
args = { step_size = 20, gamma = 0.5 }
```

### 蒸馏参数调优建议

- **temperature**: 
  - 较高（4-8）：软标签更平滑，传递更多类别间相似性信息
  - 较低（1-2）：更接近硬标签，训练更稳定
  
- **alpha**:
  - 0.3-0.5：平衡硬标签和软标签，推荐值
  - 0.0：纯软标签蒸馏（需要教师网络质量很高）
  - 1.0：纯硬标签训练（退化为普通监督学习）

## 输出结果

蒸馏完成后，会得到：

1. **模型文件**：
   - `best_student_X.X.X-distill.pth`: 最佳学生网络
   - `final_student_X.X.X-distill.pth`: 最终学生网络

2. **规则文件**：
   - `extracted_rule.json`: JSON格式的完整规则表

3. **控制台输出**：
   ```
   B3/S23
   
   Birth conditions: [(3, 0.98), ...]
   Survival conditions: [(2, 0.97), (3, 0.99), ...]
   
   Full Rule Table:
   Cell State   Neighbors  Prediction   Prob    
   --------------------------------------------
   0            0          0            0.001
   0            1          0            0.002
   0            2          0            0.005
   0            3          1            0.984    <-- Birth
   ...
   ```

## 学生网络架构细节

```python
InterpretableCA(
    (counter): Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
    # 权重固定为（归一化后）:
    # [[0.125, 0.125, 0.125],
    #  [0.125, 0.000, 0.125],
    #  [0.125, 0.125, 0.125]]
    
    (decision): Sequential(
        (0): Conv2d(2, 16, kernel_size=1)  # 输入: [cell_state, neighbor_count]
        (1): ReLU()
        (2): Conv2d(16, 2, kernel_size=1)  # 输出: [dead_logit, alive_logit]
    )
)
```

## 科学发现流程

这个方法对应了完整的科学发现流程：

1. **探索阶段**：教师网络从数据中学习黑盒表示，通过卷积核可视化发现Moore邻域结构
2. **建模阶段**：设计结构受限的学生网络，强制先验知识（计数+决策）
3. **蒸馏阶段**：将教师知识迁移到学生网络，保持性能的同时获得可解释性
4. **提取阶段**：从学生网络直接读取显式规则（B/S字符串）

## 注意事项

1. **教师网络质量**：蒸馏效果依赖于教师网络的准确性，确保教师网络已充分训练
2. **数据分布**：如果某些邻居计数组合在训练数据中极少出现，学生网络可能无法准确学习对应规则
3. **隐藏层维度**：`hidden_dim`较小时（如8-16），决策层更可解释，但可能损失一些拟合能力

## 扩展思路

- **多规则蒸馏**：可同时训练多个学生网络，每个对应不同的CA规则
- **连续CA**：将输入从{0,1}改为[0,1]，学生网络仍可工作，而直接统计方法失效
- **噪声鲁棒性**：教师网络在噪声数据上训练，学生网络继承其鲁棒性
