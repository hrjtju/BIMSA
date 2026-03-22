import re

import numpy as np
import matplotlib.pyplot as plt

LTL_FILENAME_PATTERN = re.compile(
    r'R(?P<radius>\d+)'
    r'_C(?P<states>\d+)'
    r'_M(?P<middle>[01])'
    r'_S(?P<smin>\d+)-(?P<smax>\d+)'
    r'_B(?P<bmin>\d+)-(?P<bmax>\d+)'
    r'_(?P<neighborhood>N[BCDMNX2+#])'
)

def generate_neighborhood_matrix(neighbor_type, radius, center=True):
    """
    生成邻域矩阵
    
    参数:
        neighbor_type: 邻域类型，可以是全名或缩写
            - "Moore" 或 "NM": 摩尔邻域 (方形)
            - "von Neumann" 或 "NN": 冯·诺依曼邻域 (菱形)
            - "Circular" 或 "NC": 圆形邻域
            - "L2"/"Euclidean" 或 "N2": L2/欧几里得邻域 (同圆形)
            - "Checkerboard" 或 "NB": 棋盘格邻域
            - "Aligned Checkerboard" 或 "ND": 对齐棋盘格邻域
            - "Cross" 或 "N+": 十字形邻域
            - "Saltire" 或 "NX": 斜十字/X形邻域
            - "Star" 或 "N*": 星形邻域 (十字+X形)
            - "Hash" 或 "N#": 井字形邻域
        radius: 邻域半径 (整数)
    
    返回:
        numpy.ndarray: 尺寸为 (2*radius+1, 2*radius+1) 的0-1矩阵
    """
    if radius < 0:
        raise ValueError("半径必须是非负整数")
    
    size_ = 2 * radius + 1
    # 创建坐标网格，中心在 (0,0)
    # y是行(垂直), x是列(水平)
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    
    # 用于棋盘格类型的网格索引 (0到size-1)
    i, j = np.ogrid[0:size_, 0:size_]
    
    neighbor_type = neighbor_type.strip()
    
    if neighbor_type in ["Moore", "NM"]:
        # 摩尔邻域: 填满整个方形
        matrix = np.ones((size_, size_), dtype=int)
        
    elif neighbor_type in ["von Neumann", "NN"]:
        # 冯·诺依曼邻域: 曼哈顿距离 <= 半径
        matrix = (np.abs(x) + np.abs(y) <= radius).astype(int)
        
    elif neighbor_type in ["Circular", "NC", "N2"]:
        # 圆形邻域: 欧几里得距离 <= 半径
        matrix = (x**2 + y**2 <= radius**2).astype(int)
        
    elif neighbor_type in ["Checkerboard", "NB"]:
        # 棋盘格邻域: (i+j) % 2 == 0，中心对齐
        # 中心点 (radius, radius) 应该为 1 (黄色)
        matrix = (((i - radius) + (j - radius)) % 2 == 0).astype(int)
        
    elif neighbor_type in ["Aligned Checkerboard", "ND"]:
        # 对齐棋盘格: 相对于NB偏移一个相位
        # 或者可以理解为 (i % 2 == 0) & (j % 2 == 0) 的变体
        # 这里采用与NB错开相位的实现
        matrix = (((i - radius) + (j - radius)) % 2 == 1).astype(int)
        
    elif neighbor_type in ["Cross", "N+"]:
        # 十字形邻域: x==0 或 y==0
        matrix = ((x == 0) | (y == 0)).astype(int)
        
    elif neighbor_type in ["Saltire", "NX"]:
        # 斜十字/X形邻域: |x| == |y|
        matrix = (np.abs(x) == np.abs(y)).astype(int)
        
    elif neighbor_type in ["Star", "N*"]:
        # 星形邻域: 十字 + X形
        matrix = ((x == 0) | (y == 0) | (np.abs(x) == np.abs(y))).astype(int)
        
    elif neighbor_type in ["Hash", "N#"]:
        # 井字形邻域: 十字形 + 特定位置的点形成"#"形状
        # 根据图示，是在十字基础上，在臂的两侧添加点
        matrix = (np.abs(x) == 1) | (np.abs(y) == 1)
        
    else:
        raise ValueError(f"未知的邻域类型: {neighbor_type}")
    
    
    matrix[matrix.shape[0]//2, matrix.shape[0]//2] = True if center else False
    
    return matrix.astype(np.int32)


def visualize_matrix(matrix, title="", figsize=(5, 5)):
    """
    可视化邻域矩阵
    黑色=0, 黄色=1, 红色十字标出中心点
    """
    plt.figure(figsize=figsize)
    # 使用黄色和黑色显示
    plt.imshow(matrix, cmap='cividis', interpolation='nearest')
    plt.colorbar(label='Value (0=Black, 1=Yellow)')
    
    # 标出中心点
    center = matrix.shape[0] // 2
    plt.plot(center, center, 'r+', markersize=15, markeredgewidth=2, label='Center')
    
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


# 示例用法
if __name__ == "__main__":
    radius = 3  # 你可以修改半径
    
    # 测试所有类型
    types = [
        ("Moore", "NM"),
        ("von Neumann", "NN"),
        ("Circular", "NC"),
        ("L2", "N2"),
        ("Checkerboard", "NB"),
        ("Aligned Checkerboard", "ND"),
        ("Cross", "N+"),
        ("Saltire", "NX"),
        ("Star", "N*"),
        ("Hash", "N#")
    ]
    
    print(f"生成半径为 {radius} 的邻域矩阵 (尺寸: {2*radius+1}x{2*radius+1})")
    print("=" * 60)
    
    for full_name, abbr in types:
        mat = generate_neighborhood_matrix(full_name, radius)
        # 或者使用缩写: mat = generate_neighborhood_matrix(abbr, radius)
        
        print(f"\n{full_name} ({abbr}):")
        print(mat)
        
        # 统计1的个数
        print(f"  1的个数 (邻域大小): {np.sum(mat)}")
        
        # 可视化 (取消注释以显示图形)
        # visualize_matrix(mat, f"{full_name} ({abbr}), r={radius}")

