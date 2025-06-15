import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from numba import njit, prange
import time

# 参数设置
L = 64.0
rho = 3.0
N = int(rho*L**2)
print("N:", N)

r0 = 1.0
delta_t = 1.0
factor = 0.5
v0 = r0 / delta_t * factor
eta = 0.15
grid_size = r0  # 网格大小设为相互作用半径
grid_n = int(L / grid_size)

# 初始条件
pos = np.random.uniform(0, L, size=(N, 2))
orient = np.random.uniform(-np.pi, np.pi, size=N)

# 创建网格数据结构
head = np.full((grid_n, grid_n), -1, dtype=np.int32)
next_particle = np.full(N, -1, dtype=np.int32)

# ===========================================
# 使用Numba优化的核心计算函数
# ===========================================
@njit
def build_grid(pos, head, next_particle, L, grid_size, grid_n):
    """构建网格邻接表"""
    head.fill(-1)
    next_particle.fill(-1)
    
    for i in range(len(pos)):
        x, y = pos[i]
        # 计算网格索引（周期性边界处理）
        grid_x = int(x / grid_size) % grid_n
        grid_y = int(y / grid_size) % grid_n
        
        # 将粒子插入网格链表
        next_particle[i] = head[grid_x, grid_y]
        head[grid_x, grid_y] = i

@njit(parallel=True)
def calculate_orientation(orient, pos, head, next_particle, L, r0, eta, grid_n, grid_size):
    """计算新的粒子方向"""
    new_orient = np.empty_like(orient)
    inv_r0_2 = 1.0 / (r0 * r0)
    
    for idx in prange(len(pos)):
        x, y = pos[idx]
        grid_x = int(x / grid_size) % grid_n
        grid_y = int(y / grid_size) % grid_n
        
        sx, sy = 0.0, 0.0
        count = 0
        
        # 检查3x3的相邻网格
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                # 考虑周期性边界
                nbr_x = (grid_x + dx) % grid_n
                nbr_y = (grid_y + dy) % grid_n
                
                p = head[nbr_x, nbr_y]
                while p >= 0:
                    if p != idx:  # 避免与自己比较
                        dx_pos = pos[p, 0] - x
                        dy_pos = pos[p, 1] - y
                        
                        # 周期性边界校正
                        if dx_pos > L/2: dx_pos -= L
                        elif dx_pos < -L/2: dx_pos += L
                        if dy_pos > L/2: dy_pos -= L
                        elif dy_pos < -L/2: dy_pos += L
                        
                        dist_sq = dx_pos*dx_pos + dy_pos*dy_pos
                        
                        if dist_sq < r0*r0:
                            sx += np.cos(orient[p])
                            sy += np.sin(orient[p])
                            count += 1
                    p = next_particle[p]
        
        # 计算平均方向（含噪声）
        avg_theta = np.arctan2(sy, sx) if count > 0 else orient[idx]
        noise = eta * (np.random.uniform() * 2*np.pi - np.pi)
        new_orient[idx] = avg_theta + noise
        
    return new_orient

@njit(parallel=True)
def update_positions(pos, orient, v0, L):
    """更新粒子位置"""
    for i in prange(len(pos)):
        pos[i, 0] = (pos[i, 0] + np.cos(orient[i]) * v0) % L
        pos[i, 1] = (pos[i, 1] + np.sin(orient[i]) * v0) % L
# ===========================================

# 绘图设置
fig = plt.figure(figsize=(12, 6))
ax = plt.subplot(1, 2, 1)
ax1 = plt.subplot(1, 2, 2)

plt.suptitle("Numba加速的Vicsek模型模拟")
ax.set_title("粒子运动")
ax1.set_title("有序参数")
ax.set_xlim(0, L)
ax.set_ylim(0, L)

qv = ax.quiver(pos[:, 0], pos[:, 1], 
               np.cos(orient), np.sin(orient), orient, 
               clim=[-np.pi, np.pi], scale=20)

ord_ls = []
order_plot, = ax1.plot([], [], lw=2, label='平均有序参数')
ax1.set_ylim(0, 1)
ax1.set_xlim(0, 2000)
ax1.legend(loc='upper right')
ax1.set_xlabel("迭代次数")
ax1.set_ylabel("有序参数")

stats_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12,
                     verticalalignment='top', 
                     bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

# ===========================================
# 核心优化：使用网格法加速的动画更新函数
# ===========================================
def animate(i):
    global pos, orient
    
    # 1. 构建网格数据结构
    build_grid(pos, head, next_particle, L, grid_size, grid_n)
    
    # 2. 计算新方向
    start_time = time.time()
    orient = calculate_orientation(orient, pos, head, next_particle, 
                                  L, r0, eta, grid_n, grid_size)
    
    # 3. 更新位置
    update_positions(pos, orient, v0, L)
    calc_time = time.time() - start_time
    
    # 4. 更新绘图数据
    cos, sin = np.cos(orient), np.sin(orient)
    avg_orient = np.mean(orient)
    std_orient = np.std(orient)
    
    # 计算有序参数
    order_param = np.abs(np.mean(np.exp(1j * orient)))
    ord_ls.append(order_param)
    
    # 更新统计信息
    stats_text.set_text(f'{i:>5d}次迭代 | 方向: {avg_orient:.3f}±{std_orient:.3f}\n'
                       f'计算时间: {calc_time*1000:.1f}ms\n'
                       f'邻居搜索: 网格法({grid_n}x{grid_n})')
    
    # 更新图形
    qv.set_offsets(pos)
    qv.set_UVC(cos, sin, orient)
    order_plot.set_data(np.arange(len(ord_ls)), ord_ls)
    
    return qv, order_plot, stats_text
# ===========================================

# 创建并保存动画
anim = FuncAnimation(fig, animate, frames=2000, interval=1, blit=True)
plt.tight_layout()
plt.show()

# FFwriter = FFMpegWriter(fps=10, bitrate=10000)
# anim.save('vicsek_numba.mp4', writer=FFwriter)
