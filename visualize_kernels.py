import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define the convolution kernels
kernels = np.array([[[[ 0.02697355,  0.02264993,  0.02316411],
         [ 0.05246889, -0.05315712,  0.03727428],
         [ 0.05759314,  0.07870849,  0.05527936]],

        [[-1.4669605 , -1.4005249 , -1.4798267 ],
         [-1.4768263 , -0.93933254, -1.472745  ],
         [-1.504763  , -1.4481417 , -1.5047762 ]]],


       [[[-0.0411509 , -0.04258763, -0.02910726],
         [-0.0697557 ,  0.1001616 , -0.07216603],
         [-0.09008313, -0.08381517, -0.08917644]],

        [[-0.9510107 ,  0.00615819, -0.68087417],
         [ 0.14507206,  4.8816576 ,  0.3754919 ],
         [-0.5132424 ,  0.94791317, -0.28709134]]]], dtype=np.float32)

print(f"Kernels shape: {kernels.shape}")
print(f"Number of kernels: {kernels.shape[0]}")
print(f"Number of channels per kernel: {kernels.shape[1]}")
print(f"Kernel size: {kernels.shape[2]}x{kernels.shape[3]}")

# Create a figure with subplots for each kernel and channel
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle('Convolution Kernels Visualization', fontsize=16, fontweight='bold')

# Set color map - using diverging colormap to show positive/negative values
cmap = 'RdBu_r'

for kernel_idx in range(kernels.shape[0]):
    for channel_idx in range(kernels.shape[1]):
        ax = axes[kernel_idx, channel_idx]
        kernel_data = kernels[kernel_idx, channel_idx]
        
        # Create heatmap
        im = ax.imshow(kernel_data, cmap=cmap, vmax=5, vmin=-3)
        
        # Add values as text annotations
        for i in range(kernel_data.shape[0]):
            for j in range(kernel_data.shape[1]):
                text = ax.text(j, i, f'{kernel_data[i, j]:.3f}',
                             ha="center", va="center", color="black", fontsize=9)
        
        # Set title and labels
        ax.set_title(f'Kernel {kernel_idx + 1}, Channel {channel_idx + 1}', 
                    fontweight='bold', pad=10)
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels([0, 1, 2])
        ax.set_yticklabels([0, 1, 2])
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Weight Value', rotation=270, labelpad=15)

plt.tight_layout()
plt.savefig('convolution_kernels_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics for each kernel
print("\n=== Kernel Statistics ===")
for kernel_idx in range(kernels.shape[0]):
    print(f"\nKernel {kernel_idx + 1}:")
    for channel_idx in range(kernels.shape[1]):
        channel_data = kernels[kernel_idx, channel_idx]
        print(f"  Channel {channel_idx + 1}:")
        print(f"    Min: {np.min(channel_data):.4f}")
        print(f"    Max: {np.max(channel_data):.4f}")
        print(f"    Mean: {np.mean(channel_data):.4f}")
        print(f"    Std: {np.std(channel_data):.4f}")
