import time
import numpy as np
import scipy as sp
from scipy import sparse
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from itertools import count

L = 64.0
rho = 1.0
N = int(rho*L**2)
print(" N",N)
 
r0 = 1.0
delta_t = 1.0
factor = 0.5
v0 = r0 / delta_t * factor
iterations = 10000
eta = 0.15

np.random.seed(int(time.time()))
 
pos = np.random.uniform(0,L,size=(N,2))
orient = np.random.uniform(-np.pi, np.pi,size=N)

fig = plt.figure(figsize=(12,6))
ax= plt.subplot(1, 2, 1)
ax1 = plt.subplot(1, 2, 2)

plt.suptitle("Simulation of Vicsek Model and Order Parameter")
ax.set_title("Vicsek model")
ax1.set_title("Order parameter")

ord_ls = []

# plot a 2D view of arrows 
# (X, Y, U, V, C)
qv = ax.quiver(pos[:,0], pos[:,1], np.cos(orient), np.sin(orient), orient, clim=[-np.pi, np.pi])
stats_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12,
                     verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
order_plot, = ax1.plot([], [], lw=2, label='avg normalized velocity')
ax1.set_ylim(0, 1)
ax1.set_xlim(0, 2000)

def animate(i):
 
    global orient
    tree = cKDTree(pos,boxsize=[L,L])
    dist = tree.sparse_distance_matrix(tree, max_distance=r0,output_type='coo_matrix')
 
    #important 3 lines: we evaluate a quantity for every column j
    data = np.exp(orient[dist.col]*1j)
    # construct  a new sparse marix with entries in the same places ij of the dist matrix
    neigh = sparse.coo_matrix((data,(dist.row,dist.col)), shape=dist.get_shape())
    # and sum along the columns (sum over j)
    S = np.squeeze(np.asarray(neigh.tocsr().sum(axis=1)))
     
    orient = np.angle(S)+eta*np.random.uniform(-np.pi, np.pi, size=N)
 
    cos, sin= np.cos(orient), np.sin(orient)
    pos[:,0] += cos*v0
    pos[:,1] += sin*v0
 
    pos[pos>L] -= L
    pos[pos<0] += L
    
    # calculate the average orientation and show on the plot in text
    stats_text.set_text(f'{i:>5d}Its | Avg: {np.mean(orient):.3f} rad, Std: {np.std(orient):.3f} rad')
    
    
    # update average normalized velocity
    ord_ls.append(float(np.abs(np.mean(np.exp(1j*orient)))))
    print(f'{i:>4d} | Avg: {np.mean(orient):.3f} rad, Std: {np.std(orient):.3f} rad, Ord: {ord_ls[-1]:.3f}')
    order_plot.set_data(np.arange(len(ord_ls)), ord_ls)
    
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    
    qv.set_offsets(pos)
    qv.set_UVC(cos, sin,orient)
    return stats_text, qv, order_plot
 
anim = FuncAnimation(fig, animate, range(10000), interval=1, blit=True)

FFwriter = FFMpegWriter(fps=20)
anim.save('animation.mp4', writer = FFwriter)

plt.show()
