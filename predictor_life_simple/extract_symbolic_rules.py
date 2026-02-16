"""
使用 SymPy 从 state_dict 中提取显式的符号规则

将神经网络的权重转换为可读的数学表达式，支持权重稀疏化
"""

import torch
import numpy as np
import re
from collections import OrderedDict
from typing import Dict, Tuple, List
import sympy as sp
from sympy import symbols, Matrix, Piecewise, Max, exp, log, simplify, Rational, latex


def parse_tensor_string(s: str):
    """解析 tensor 字符串"""
    import ast
    s = s.strip()
    if s.endswith(')'):
        s = s[:-1]
    if s.endswith(']'):
        s = s[:-1]
    s = s.replace('tensor(', '').replace(')', '')
    try:
        array = ast.literal_eval(s)
        return np.array(array, dtype=np.float32)
    except:
        return None


def load_state_dict_from_text(file_path: str) -> OrderedDict:
    """从文本文件加载 state_dict"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    state_dict = OrderedDict()
    lines = content.split('\n')
    current_key = None
    current_tensor_str = []
    in_tensor = False
    
    for line in lines:
        line = line.strip()
        
        if line.startswith("('") and "', tensor(" in line:
            if current_key and current_tensor_str:
                tensor_str = '\n'.join(current_tensor_str)
                tensor = parse_tensor_string(tensor_str)
                if tensor is not None:
                    state_dict[current_key] = tensor
            
            match = re.match(r"\('([^']+)'\,\s*tensor\(", line)
            if match:
                current_key = match.group(1)
                tensor_start = line.find("tensor(") + 7
                current_tensor_str = [line[tensor_start:]]
                in_tensor = True
        
        elif in_tensor:
            current_tensor_str.append(line)
            if line.rstrip().endswith('))'):
                tensor_str = '\n'.join(current_tensor_str)
                tensor = parse_tensor_string(tensor_str)
                if tensor is not None:
                    state_dict[current_key] = tensor
                current_key = None
                current_tensor_str = []
                in_tensor = False
    
    if current_key and current_tensor_str:
        tensor_str = '\n'.join(current_tensor_str)
        tensor = parse_tensor_string(tensor_str)
        if tensor is not None:
            state_dict[current_key] = tensor
    
    return state_dict


def load_state_dict(file_path: str) -> OrderedDict:
    """加载 state_dict"""
    if file_path.endswith('.pth') or file_path.endswith('.pt'):
        return torch.load(file_path, map_location='cpu')
    else:
        return load_state_dict_from_text(file_path)


def prune_small_coefficients(expr: sp.Expr, threshold: float = 0.01) -> sp.Expr:
    """
    删除表达式中系数绝对值小于 threshold 的项
    
    Args:
        expr: SymPy 表达式
        threshold: 系数阈值，小于此值的项将被删除
    
    Returns:
        简化后的表达式
    """
    if expr.is_Add:
        # 处理加法表达式：过滤每一项
        new_args = []
        for term in expr.args:
            # 获取该项的数值系数
            coeff = term.as_coeff_mul()[0]
            if abs(float(coeff)) >= threshold:
                new_args.append(term)
        if len(new_args) == 0:
            return sp.Integer(0)
        elif len(new_args) == 1:
            return new_args[0]
        else:
            return sp.Add(*new_args)
    elif expr.is_Mul:
        # 处理乘法表达式（如系数 * Max(...)）
        coeff = expr.as_coeff_mul()[0]
        if abs(float(coeff)) < threshold:
            return sp.Integer(0)
    elif expr.is_Function:
        # 处理函数（如 Max, ReLU 等）
        # 递归处理参数
        new_args = [prune_small_coefficients(arg, threshold) for arg in expr.args]
        return expr.func(*new_args)
    
    return expr


class SymbolicHead:
    """
    表示一个 head 的符号计算单元
    
    结构：
    - 输入: c (cell state), n (neighbor count)
    - 隐藏层: h_i = ReLU(w1[i,0]*c + w1[i,1]*n + b1[i])
    - 输出: logits = w2 @ h + b2
    """
    
    def __init__(self, head_idx: int, w1: np.ndarray, b1: np.ndarray, 
                 w2: np.ndarray, b2: np.ndarray, threshold: float = 0.0):
        """
        Args:
            head_idx: head 索引
            w1: 第一层权重 [hidden_dim, 2]
            b1: 第一层偏置 [hidden_dim]
            w2: 第二层权重 [2, hidden_dim]
            b2: 第二层偏置 [2]
            threshold: 稀疏化阈值，绝对值小于此值的权重设为0
        """
        self.head_idx = head_idx
        
        # 应用稀疏化
        w1 = w1.copy()
        b1 = b1.copy()
        w2 = w2.copy()
        b2 = b2.copy()
        
        if threshold > 0:
            w1[np.abs(w1) < threshold] = 0
            w2[np.abs(w2) < threshold] = 0
        
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2
        
        # 统计非零参数
        self.num_nonzero = (np.count_nonzero(w1) + np.count_nonzero(b1) + 
                           np.count_nonzero(w2) + np.count_nonzero(b2))
        self.total_params = w1.size + b1.size + w2.size + b2.size
        
        # 定义符号变量
        self.c, self.n = symbols('c n', real=True)
        
    def get_symbolic_expressions(self, prune_threshold: float = 0.01) -> Tuple[sp.Expr, sp.Expr]:
        """
        获取符号表达式
        
        Args:
            prune_threshold: 剪枝阈值，系数小于此值的项将被删除
        
        Returns:
            (logits_dead, logits_alive): 两个符号表达式
        """
        # 计算隐藏层
        h_exprs = []
        for i in range(len(self.b1)):
            # 线性组合（剪枝小权重）
            w_c = self.w1[i, 0] if abs(self.w1[i, 0]) >= prune_threshold else 0
            w_n = self.w1[i, 1] if abs(self.w1[i, 1]) >= prune_threshold else 0
            b = self.b1[i] if abs(self.b1[i]) >= prune_threshold else 0
            
            linear = w_c * self.c + w_n * self.n + b
            
            # ReLU: Max(0, linear)
            if w_c == 0 and w_n == 0 and b == 0:
                h_exprs.append(0)
            else:
                h_exprs.append(Max(0, linear))
        
        # 计算输出层
        logits = []
        for j in range(2):  # dead 和 alive
            # 偏置项
            expr = self.b2[j] if abs(self.b2[j]) >= prune_threshold else 0
            
            for i in range(len(self.b1)):
                if abs(self.w2[j, i]) >= prune_threshold and h_exprs[i] != 0:
                    expr += self.w2[j, i] * h_exprs[i]
            
            # 简化并剪枝
            expr = simplify(expr)
            expr = prune_small_coefficients(expr, prune_threshold)
            logits.append(expr)
        
        return logits[0], logits[1]  # (dead, alive)
    
    def get_decision_function(self, prune_threshold: float = 0.01) -> sp.Expr:
        """
        获取决策函数：logits_alive - logits_dead
        正值表示预测 alive，负值表示预测 dead
        
        Args:
            prune_threshold: 剪枝阈值，系数小于此值的项将被删除
        """
        logit_dead, logit_alive = self.get_symbolic_expressions(prune_threshold)
        decision = simplify(logit_alive - logit_dead)
        decision = prune_small_coefficients(decision, prune_threshold)
        return decision
    
    def get_latex(self, prune_threshold: float = 0.01) -> str:
        """获取 LaTeX 格式的表达式"""
        decision = self.get_decision_function(prune_threshold)
        return latex(decision)
    
    def evaluate(self, cell_state: int, neighbor_count: int) -> Tuple[float, float]:
        """数值计算给定输入的输出"""
        c_val = float(cell_state)
        n_val = float(neighbor_count)
        
        # 隐藏层
        h = np.maximum(0, self.w1 @ np.array([c_val, n_val]) + self.b1)
        
        # 输出层
        logits = self.w2 @ h + self.b2
        
        return logits[0], logits[1]  # (dead, alive)
    
    def __str__(self, prune_threshold: float = 0.01) -> str:
        """字符串表示"""
        logit_dead, logit_alive = self.get_symbolic_expressions(prune_threshold)
        decision = self.get_decision_function(prune_threshold)
        
        s = f"Head {self.head_idx} ({self.num_nonzero}/{self.total_params} non-zero):\n"
        s += f"  Logits_dead: {logit_dead}\n"
        s += f"  Logits_alive: {logit_alive}\n"
        s += f"  Decision (alive - dead): {decision}\n"
        return s


class SymbolicEnsemble:
    """符号集成分类器"""
    
    def __init__(self, state_dict: OrderedDict, threshold: float = 0.0):
        """
        Args:
            state_dict: 模型 state_dict
            threshold: 稀疏化阈值
        """
        self.threshold = threshold
        
        # 加载 fusion_weight
        self.fusion_weight = state_dict['fusion_weight']
        if isinstance(self.fusion_weight, torch.Tensor):
            self.fusion_weight = self.fusion_weight.numpy()
        
        # softmax 归一化
        self.fusion_softmax = np.exp(self.fusion_weight) / np.sum(np.exp(self.fusion_weight))
        
        # 加载所有 heads
        self.heads = []
        head_indices = []
        for key in state_dict.keys():
            match = re.match(r'heads\.(\d+)\.0\.weight', key)
            if match:
                head_indices.append(int(match.group(1)))
        
        for head_idx in head_indices:
            prefix = f'heads.{head_idx}'
            w1 = state_dict[f'{prefix}.0.weight']
            b1 = state_dict[f'{prefix}.0.bias']
            w2 = state_dict[f'{prefix}.2.weight']
            b2 = state_dict[f'{prefix}.2.bias']
            
            if isinstance(w1, torch.Tensor):
                w1 = w1.numpy()
                b1 = b1.numpy()
                w2 = w2.numpy()
                b2 = b2.numpy()
            
            # 调整形状
            w1 = w1.squeeze().reshape(-1, 2)
            w2 = w2.squeeze().reshape(2, -1)
            
            head = SymbolicHead(head_idx, w1, b1, w2, b2, threshold)
            self.heads.append(head)
        
        self.c, self.n = symbols('c n', real=True)
    
    def get_symbolic_decision(self, prune_threshold: float = 0.01) -> sp.Expr:
        """获取集成的符号决策函数
        
        Args:
            prune_threshold: 剪枝阈值，系数小于此值的项将被删除
        """
        ensemble_expr = 0
        
        for head, weight in zip(self.heads, self.fusion_softmax):
            if weight >= prune_threshold:  # 也过滤小的融合权重
                decision = head.get_decision_function(prune_threshold)
                if decision != 0:
                    ensemble_expr += weight * decision
        
        # 简化和剪枝
        ensemble_expr = simplify(ensemble_expr)
        ensemble_expr = prune_small_coefficients(ensemble_expr, prune_threshold)
        return ensemble_expr
    
    def get_piecewise_rule(self, threshold: float = 0.0) -> Dict[Tuple[int, int], str]:
        """
        获取分段规则表
        
        Returns:
            字典：{(cell_state, neighbor_count): 'alive'/'dead'}
        """
        rules = {}
        
        for cell_state in [0, 1]:
            for neighbor_count in range(9):
                # 计算所有 heads 的加权决策
                decision_val = 0
                for head, weight in zip(self.heads, self.fusion_softmax):
                    _, logit_alive = head.evaluate(cell_state, neighbor_count)
                    _, logit_dead = head.evaluate(cell_state, neighbor_count)
                    decision_val += weight * (logit_alive - logit_dead)
                
                prediction = 'alive' if decision_val > threshold else 'dead'
                rules[(cell_state, neighbor_count)] = prediction
        
        return rules
    
    def extract_bs_rule(self) -> Tuple[List[int], List[int]]:
        """提取 B/S 规则"""
        rules = self.get_piecewise_rule()
        
        birth = []
        survival = []
        
        for neighbor_count in range(9):
            # Birth: cell=0 -> alive
            if rules[(0, neighbor_count)] == 'alive':
                birth.append(neighbor_count)
            # Survival: cell=1 -> alive
            if rules[(1, neighbor_count)] == 'alive':
                survival.append(neighbor_count)
        
        return birth, survival
    
    def print_detailed_analysis(self, prune_threshold: float = 0.01):
        """打印详细分析
        
        Args:
            prune_threshold: 表达式剪枝阈值
        """
        print("=" * 80)
        print("SYMBOLIC NEURAL NETWORK ANALYSIS")
        print(f"(Pruning threshold: {prune_threshold})")
        print("=" * 80)
        
        # Fusion weights
        print("\n1. FUSION WEIGHTS:")
        print(f"   Raw: {self.fusion_weight}")
        print(f"   Softmax: {self.fusion_softmax}")
        
        # Individual heads
        print("\n2. INDIVIDUAL HEADS:")
        for head in self.heads:
            print(head.__str__(prune_threshold))
        
        # Ensemble decision
        print("\n3. ENSEMBLE DECISION FUNCTION:")
        ensemble = self.get_symbolic_decision(prune_threshold)
        print(f"   f(c, n) = {ensemble}")
        print(f"\n   LaTeX: {latex(ensemble)}")
        
        # Rule table
        print("\n4. DECISION TABLE:")
        print("   " + "-" * 50)
        print(f"   {'Cell':<6} {'Neighbors':<10} {'Decision':<12} {'Prediction':<10}")
        print("   " + "-" * 50)
        
        ensemble_func = self.get_symbolic_decision()
        for cell_state in [0, 1]:
            for neighbor_count in range(9):
                # 数值评估
                val = sum(
                    weight * (head.evaluate(cell_state, neighbor_count)[1] - 
                             head.evaluate(cell_state, neighbor_count)[0])
                    for head, weight in zip(self.heads, self.fusion_softmax)
                )
                pred = 'ALIVE' if val > 0 else 'dead'
                marker = '*' if abs(val) < 0.5 else ' '
                print(f"   {cell_state:<6} {neighbor_count:<10} {val:>+10.4f}   {marker}{pred:<10}")
        
        print("   " + "-" * 50)
        print("   (* indicates uncertain prediction)")
        
        # Extracted rule
        birth, survival = self.extract_bs_rule()
        print(f"\n5. EXTRACTED RULE: B{','.join(map(str, birth))}/S{','.join(map(str, survival))}")
        
        # 与 B3/S23 对比
        std_birth = {3}
        std_survival = {2, 3}
        actual_birth = set(birth)
        actual_survival = set(survival)
        
        print(f"\n6. COMPARISON WITH B3/S23:")
        print(f"   Birth: Match={actual_birth == std_birth}, "
              f"Expected={std_birth}, Got={actual_birth}, "
              f"Diff={actual_birth ^ std_birth}")
        print(f"   Survival: Match={actual_survival == std_survival}, "
              f"Expected={std_survival}, Got={actual_survival}, "
              f"Diff={actual_survival ^ std_survival}")
        
        print("=" * 80)
    
    def generate_python_code(self) -> str:
        """生成纯 Python 实现代码"""
        code = '''"""
Auto-generated from symbolic extraction
Original: Neural Network with {} heads
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
    fusion_weights = [{}]
    
    # Heads
    decisions = []
'''.format(len(self.heads), ', '.join(f'{w:.6f}' for w in self.fusion_softmax))
        
        for i, head in enumerate(self.heads):
            code += f"\n    # Head {i}\n"
            code += f"    # W1 shape: {head.w1.shape}, W2 shape: {head.w2.shape}\n"
            code += f"    h{i} = np.maximum(0, np.array([\n"
            for j in range(len(head.b1)):
                w_c = head.w1[j, 0]
                w_n = head.w1[j, 1]
                b = head.b1[j]
                if w_c != 0 or w_n != 0 or b != 0:
                    code += f"        {w_c:.6f}*c + {w_n:.6f}*n + {b:.6f},\n"
                else:
                    code += f"        0.0,\n"
            code += f"    ]))\n"
            
            code += f"    logits{i} = np.array([\n"
            for j in range(2):
                expr_parts = [f"{head.b2[j]:.6f}"]
                for k in range(len(head.b1)):
                    if head.w2[j, k] != 0:
                        expr_parts.append(f"{head.w2[j, k]:.6f}*h{i}[{k}]")
                code += f"        {' + '.join(expr_parts)},\n"
            code += f"    ])\n"
            code += f"    decisions.append(logits{i}[1] - logits{i}[0])\n"
        
        code += '''
    # Ensemble decision
    final_decision = sum(w * d for w, d in zip(fusion_weights, decisions))
    
    return 1 if final_decision > 0 else 0

# Lookup table version (faster)
LOOKUP_TABLE = {
'''
        rules = self.get_piecewise_rule()
        for cell_state in [0, 1]:
            for neighbor_count in range(9):
                pred = 1 if rules[(cell_state, neighbor_count)] == 'alive' else 0
                code += f"    ({cell_state}, {neighbor_count}): {pred},\n"
        
        code += '''}

def predict_fast(cell_state: int, neighbor_count: int) -> int:
    """Fast version using lookup table"""
    return LOOKUP_TABLE[(cell_state, neighbor_count)]
'''
        return code


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract symbolic rules from state_dict")
    parser.add_argument("--input", type=str, required=True,
                       help="Path to state_dict file")
    parser.add_argument("--threshold", type=float, default=0.0,
                       help="Sparsification threshold for weights (set small weights to 0)")
    parser.add_argument("--prune_threshold", type=float, default=0.01,
                       help="Pruning threshold for symbolic expression coefficients (default: 0.01)")
    parser.add_argument("--output_code", type=str, default=None,
                       help="Output Python code file")
    parser.add_argument("--output_analysis", type=str, default=None,
                       help="Output analysis text file")
    args = parser.parse_args()
    
    # 加载 state_dict
    print(f"Loading state_dict from {args.input}")
    state_dict = load_state_dict(args.input)
    print(f"Loaded {len(state_dict)} parameters")
    
    # 创建符号集成模型
    print(f"\nCreating symbolic model with:")
    print(f"  - Weight sparsification threshold: {args.threshold}")
    print(f"  - Expression pruning threshold: {args.prune_threshold}")
    ensemble = SymbolicEnsemble(state_dict, threshold=args.threshold)
    
    # 打印详细分析
    ensemble.print_detailed_analysis(prune_threshold=args.prune_threshold)
    
    # 保存分析到文件
    if args.output_analysis:
        import sys
        original_stdout = sys.stdout
        with open(args.output_analysis, 'w') as f:
            sys.stdout = f
            ensemble.print_detailed_analysis(prune_threshold=args.prune_threshold)
        sys.stdout = original_stdout
        print(f"\nSaved analysis to {args.output_analysis}")
    
    # 生成 Python 代码
    if args.output_code:
        code = ensemble.generate_python_code()
        with open(args.output_code, 'w') as f:
            f.write(code)
        print(f"\nSaved Python code to {args.output_code}")


if __name__ == "__main__":
    main()
