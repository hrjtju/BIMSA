"""
符号规则提取示例脚本
展示如何使用 extract_symbolic_rules 模块
"""

from extract_symbolic_rules import SymbolicEnsemble, load_state_dict
from sympy import latex, simplify, symbols
import numpy as np


def example_basic_usage():
    """基本使用示例"""
    print("=" * 80)
    print("示例 1: 基本使用")
    print("=" * 80)
    
    # 假设 state_dict 已经加载
    # state_dict = load_state_dict('distilled.txt')
    
    # 创建符号模型
    # ensemble = SymbolicEnsemble(state_dict, threshold=0.0)
    
    print("加载模型: ensemble = SymbolicEnsemble(state_dict, threshold=0.0)")
    print("获取决策函数: decision = ensemble.get_symbolic_decision()")
    print("提取规则: birth, survival = ensemble.extract_bs_rule()")


def example_compare_thresholds(state_dict_path: str):
    """对比不同稀疏化阈值的效果"""
    print("\n" + "=" * 80)
    print("示例 2: 对比不同稀疏化阈值")
    print("=" * 80)
    
    state_dict = load_state_dict(state_dict_path)
    
    thresholds = [0.0, 0.05, 0.1, 0.2]
    
    print(f"{'Threshold':<12} {'Non-zero Params':<20} {'Extracted Rule':<20} {'Correct'}")
    print("-" * 80)
    
    for threshold in thresholds:
        ensemble = SymbolicEnsemble(state_dict, threshold=threshold)
        
        # 统计非零参数
        total_nonzero = sum(head.num_nonzero for head in ensemble.heads)
        total_params = sum(head.total_params for head in ensemble.heads)
        
        # 提取规则
        birth, survival = ensemble.extract_bs_rule()
        rule_str = f"B{','.join(map(str, birth))}/S{','.join(map(str, survival))}"
        
        # 检查是否正确（假设目标是 B3/S23）
        is_correct = (set(birth) == {3} and set(survival) == {2, 3})
        
        print(f"{threshold:<12} {total_nonzero}/{total_params:<12} {rule_str:<20} {'✓' if is_correct else '✗'}")


def example_analyze_single_head(state_dict_path: str):
    """分析单个 head 的符号表达式"""
    print("\n" + "=" * 80)
    print("示例 3: 分析单个 Head 的符号表达式")
    print("=" * 80)
    
    state_dict = load_state_dict(state_dict_path)
    ensemble = SymbolicEnsemble(state_dict, threshold=0.05)
    
    # 分析第一个 head
    head = ensemble.heads[0]
    
    print(f"\nHead 0 统计:")
    print(f"  非零参数: {head.num_nonzero}/{head.total_params}")
    print(f"  稀疏度: {1 - head.num_nonzero/head.total_params:.2%}")
    
    # 获取符号表达式
    logit_dead, logit_alive = head.get_symbolic_expressions()
    decision = head.get_decision_function()
    
    print(f"\n符号表达式:")
    print(f"  Logit_dead = {logit_dead}")
    print(f"  Logit_alive = {logit_alive}")
    print(f"\n决策函数:")
    print(f"  f(c, n) = {decision}")
    
    # LaTeX 格式
    print(f"\nLaTeX 格式:")
    print(f"  {latex(decision)}")
    
    # 数值评估示例
    print(f"\n数值评估:")
    for cell_state in [0, 1]:
        for neighbor_count in [2, 3, 4]:
            logit_d, logit_a = head.evaluate(cell_state, neighbor_count)
            pred = "ALIVE" if logit_a > logit_d else "dead"
            margin = logit_a - logit_d
            print(f"  Cell={cell_state}, Neighbors={neighbor_count}: {pred} (margin={margin:+.4f})")


def example_ensemble_analysis(state_dict_path: str):
    """集成分析示例"""
    print("\n" + "=" * 80)
    print("示例 4: 集成模型分析")
    print("=" * 80)
    
    state_dict = load_state_dict(state_dict_path)
    ensemble = SymbolicEnsemble(state_dict, threshold=0.05)
    
    # 融合权重分析
    print("\n融合权重分析:")
    print(f"  原始权重: {ensemble.fusion_weight}")
    print(f"  Softmax:  {ensemble.fusion_softmax}")
    print(f"  主导 Head: {np.argmax(ensemble.fusion_softmax)} (权重={ensemble.fusion_softmax.max():.4f})")
    
    # 集成决策函数
    ensemble_decision = ensemble.get_symbolic_decision()
    print(f"\n集成决策函数 (简化前):")
    print(f"  {ensemble_decision}")
    
    # 尝试简化
    ensemble_simplified = simplify(ensemble_decision)
    print(f"\n集成决策函数 (简化后):")
    print(f"  {ensemble_simplified}")
    
    # 规则表
    print(f"\n完整规则表:")
    print(f"  {'Cell':<6} {'Neighbors':<10} {'Decision':<12} {'Prediction':<10}")
    print(f"  {'-' * 50}")
    
    for cell_state in [0, 1]:
        for neighbor_count in range(9):
            decision_val = sum(
                weight * (head.evaluate(cell_state, neighbor_count)[1] - 
                         head.evaluate(cell_state, neighbor_count)[0])
                for head, weight in zip(ensemble.heads, ensemble.fusion_softmax)
            )
            pred = "ALIVE" if decision_val > 0 else "dead"
            marker = "*" if abs(decision_val) < 0.3 else " "
            print(f"  {cell_state:<6} {neighbor_count:<10} {decision_val:>+10.4f}   {marker}{pred:<10}")


def example_generate_code(state_dict_path: str):
    """生成代码示例"""
    print("\n" + "=" * 80)
    print("示例 5: 生成纯 Python 代码")
    print("=" * 80)
    
    state_dict = load_state_dict(state_dict_path)
    ensemble = SymbolicEnsemble(state_dict, threshold=0.1)
    
    # 生成代码
    code = ensemble.generate_python_code()
    
    # 保存到文件
    output_file = 'generated_rules_example.py'
    with open(output_file, 'w') as f:
        f.write(code)
    
    print(f"代码已保存到: {output_file}")
    print(f"\n代码预览 (前 30 行):")
    print("-" * 80)
    for i, line in enumerate(code.split('\n')[:30]):
        print(f"  {line}")
    print("  ...")


def example_verify_equivalence(state_dict_path: str):
    """验证符号模型与原始模型的等价性"""
    print("\n" + "=" * 80)
    print("示例 6: 验证符号模型与原始模型等价性")
    print("=" * 80)
    
    state_dict = load_state_dict(state_dict_path)
    ensemble = SymbolicEnsemble(state_dict, threshold=0.0)
    
    print("\n对比所有 18 种输入组合:")
    print(f"  {'Cell':<6} {'Neighbors':<10} {'Symbolic':<12} {'Table':<12} {'Match'}")
    print(f"  {'-' * 60}")
    
    rules = ensemble.get_piecewise_rule()
    
    all_match = True
    for cell_state in [0, 1]:
        for neighbor_count in range(9):
            # 符号模型预测
            symbolic_decision = sum(
                weight * (head.evaluate(cell_state, neighbor_count)[1] - 
                         head.evaluate(cell_state, neighbor_count)[0])
                for head, weight in zip(ensemble.heads, ensemble.fusion_softmax)
            )
            symbolic_pred = 1 if symbolic_decision > 0 else 0
            
            # 查表预测
            table_pred = 1 if rules[(cell_state, neighbor_count)] == 'alive' else 0
            
            match = symbolic_pred == table_pred
            all_match = all_match and match
            
            marker = "✓" if match else "✗"
            print(f"  {cell_state:<6} {neighbor_count:<10} {symbolic_pred:<12} {table_pred:<12} {marker}")
    
    print(f"\n  所有预测一致: {'✓' if all_match else '✗'}")


def main():
    import sys
    
    print("符号规则提取示例")
    print("=" * 80)
    print("\n这些示例展示了如何从神经网络权重中提取显式符号规则")
    print("\n使用方法:")
    print("  python example_symbolic_extraction.py <path_to_distilled.txt>")
    
    if len(sys.argv) < 2:
        print("\n  注意: 请提供 distilled.txt 文件路径以运行完整示例")
        print("  运行基本示例...")
        example_basic_usage()
        return
    
    state_dict_path = sys.argv[1]
    
    try:
        example_basic_usage()
        example_compare_thresholds(state_dict_path)
        example_analyze_single_head(state_dict_path)
        example_ensemble_analysis(state_dict_path)
        example_generate_code(state_dict_path)
        example_verify_equivalence(state_dict_path)
        
        print("\n" + "=" * 80)
        print("所有示例运行完成!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
