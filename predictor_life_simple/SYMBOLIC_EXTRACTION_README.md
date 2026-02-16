# 符号规则提取说明

## 功能

将蒸馏后的神经网络权重转换为显式的数学表达式，支持：
- **符号表达式**：使用 SymPy 表示每个 head 的计算过程
- **权重稀疏化**：将绝对值小于阈值的权重设为 0
- **规则提取**：自动提取 B/S 规则字符串
- **代码生成**：生成纯 Python 实现（无需 PyTorch）

## 数学形式

每个 head 的计算过程：

```
输入: c (cell state ∈ {0,1}), n (neighbor count ∈ {0,...,8})

隐藏层: h_i = ReLU(w1[i,0]*c + w1[i,1]*n + b1[i])  for i=0,...,31

输出: logits = W2 @ h + b2
      logits[0] = P(dead)   的对数几率
      logits[1] = P(alive)  的对数几率

决策: alive if logits[1] > logits[0]
```

集成模型：
```
ensemble_decision = Σ softmax(fusion_weight)[i] * (logits_alive_i - logits_dead_i)
```

## 使用方法

### 1. 基本使用（无稀疏化）

```bash
cd predictor_life_simple

python extract_symbolic_rules.py \
    --input distilled.txt \
    --output_analysis analysis.txt \
    --output_code extracted_rules.py
```

### 2. 使用稀疏化（推荐）

```bash
python extract_symbolic_rules.py \
    --input distilled.txt \
    --threshold 0.1 \
    --output_analysis analysis_sparse.txt \
    --output_code extracted_rules_sparse.py
```

阈值选择建议：
- `0.0`：无稀疏化，保留所有权重
- `0.05`：轻度稀疏化，移除微小噪声
- `0.1`：中度稀疏化，显著简化表达式
- `0.2`：重度稀疏化，可能损失精度

## 输出内容

### 1. 控制台输出

```
================================================================================
SYMBOLIC NEURAL NETWORK ANALYSIS
================================================================================

1. FUSION WEIGHTS:
   Raw: [ 1.8466  0.5555  1.048  ...]
   Softmax: [0.312 0.089 0.141 ...]

2. INDIVIDUAL HEADS:
   Head 0 (45/130 non-zero):
     Logits_dead: -0.2707 - 0.10298*Max(0, 0.21481*c - 0.29545*n - 0.6571) + ...
     Logits_alive: 0.3192 + ...
     Decision (alive - dead): ...

3. ENSEMBLE DECISION FUNCTION:
   f(c, n) = 0.312*(...) + 0.089*(...) + ...
   
   LaTeX: 0.312 \left(- 0.103 \max(0, ...)

4. DECISION TABLE:
   --------------------------------------------------
   Cell   Neighbors  Decision     Prediction  
   --------------------------------------------------
   0      0           -2.3456   dead
   0      1           -1.2345   dead
   0      2           -0.5678   dead
   0      3           +1.2345 * ALIVE       <-- Birth
   ...
   1      2           +0.9876   ALIVE       <-- Survival
   1      3           +1.4567   ALIVE       <-- Survival
   --------------------------------------------------

5. EXTRACTED RULE: B3/S23

6. COMPARISON WITH B3/S23:
   Birth: Match=True, Expected={3}, Got={3}, Diff=set()
   Survival: Match=True, Expected={2, 3}, Got={2, 3}, Diff=set()
================================================================================
```

### 2. 生成的 Python 代码 (`extracted_rules.py`)

```python
"""
Auto-generated from symbolic extraction
Original: Neural Network with 8 heads
"""
import numpy as np

def predict(cell_state: int, neighbor_count: int) -> int:
    """
    Predict next state
    Returns: 0 (dead) or 1 (alive)
    """
    c = float(cell_state)
    n = float(neighbor_count)
    
    # Fusion weights (softmax normalized)
    fusion_weights = [0.312000, 0.089000, ...]
    
    # Heads
    decisions = []
    
    # Head 0
    h0 = np.maximum(0, np.array([
        0.214810*c + -0.295450*n + -0.657100,
        0.447350*c + -1.782100*n + 0.665100,
        ...
    ]))
    logits0 = np.array([
        -0.270700 + 0.102980*h0[0] + ...,
        0.319200 + -0.000334*h0[0] + ...,
    ])
    decisions.append(logits0[1] - logits0[0])
    
    # ... more heads
    
    # Ensemble decision
    final_decision = sum(w * d for w, d in zip(fusion_weights, decisions))
    
    return 1 if final_decision > 0 else 0

# Lookup table version (faster)
LOOKUP_TABLE = {
    (0, 0): 0, (0, 1): 0, (0, 2): 0, (0, 3): 1, ...
    (1, 0): 0, (1, 1): 0, (1, 2): 1, (1, 3): 1, ...
}

def predict_fast(cell_state: int, neighbor_count: int) -> int:
    """Fast version using lookup table"""
    return LOOKUP_TABLE[(cell_state, neighbor_count)]
```

## 在 Jupyter Notebook 中使用

```python
from extract_symbolic_rules import SymbolicEnsemble, load_state_dict

# 加载模型
state_dict = load_state_dict('distilled.txt')

# 创建符号模型（带稀疏化）
ensemble = SymbolicEnsemble(state_dict, threshold=0.1)

# 查看符号表达式
decision_func = ensemble.get_symbolic_decision()
print(f"决策函数: {decision_func}")

# 获取 LaTeX 格式
from sympy import latex
print(f"LaTeX: {latex(decision_func)}")

# 提取规则
birth, survival = ensemble.extract_bs_rule()
print(f"B{birth}/S{survival}")

# 生成代码
code = ensemble.generate_python_code()
with open('my_rules.py', 'w') as f:
    f.write(code)
```

## 对比不同阈值的效果

```bash
# 无稀疏化
python extract_symbolic_rules.py --input distilled.txt --threshold 0.0 \
    --output_analysis analysis_0.0.txt

# 轻度稀疏化  
python extract_symbolic_rules.py --input distilled.txt --threshold 0.05 \
    --output_analysis analysis_0.05.txt

# 中度稀疏化
python extract_symbolic_rules.py --input distilled.txt --threshold 0.1 \
    --output_analysis analysis_0.1.txt

# 比较非零参数数量
grep "non-zero" analysis_*.txt
```

## 验证生成的代码

```python
import extracted_rules

# 测试所有情况
for cell in [0, 1]:
    for neighbors in range(9):
        pred = extracted_rules.predict(cell, neighbors)
        pred_fast = extracted_rules.predict_fast(cell, neighbors)
        print(f"Cell={cell}, Neighbors={neighbors}: Predict={pred}, Fast={pred_fast}")
```
