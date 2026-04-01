from typing import List

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def erf_threshold_by_energy(erf_matrix: np.ndarray) -> List[tuple[np.ndarray, float, float]]:
    """
    Slice to get estimated effective neighborhoods based on the energy distribution of the erf_matrix.
    """
    
    center_point = np.zeros_like(erf_matrix)
    center_point[erf_matrix.shape[0]//2, erf_matrix.shape[1]//2] = True
    res = []
    seen = set()
    
    for target_energy in sorted(np.linspace(0, 1, 10), reverse=True):
        if target_energy < 1e-3 or target_energy > 1 - 1e-3:
            continue
        
        flat = erf_matrix.flatten()
        sorted_indices = np.argsort(flat)[::-1]
        sorted_values = flat[sorted_indices]
        cumsum = np.cumsum(sorted_values)
        cumsum /= cumsum[-1]
        threshold_idx = np.searchsorted(cumsum, target_energy)
        threshold = sorted_values[threshold_idx]
        mask = erf_matrix >= threshold
        
        mask = ((np.rot90(mask, 1) + np.rot90(mask, 3) + mask + np.rot90(mask, 2)) / 4 + center_point) > 0
        
        if mask.tobytes() in seen:
            continue
        else:
            seen.add(mask.tobytes())
            res.append((mask, threshold, target_energy))
    
    return res

def plot_neighborhoods(model, rule_str, neighborhoods, out_dir=None):
    # plot the mask under different energy threshold

    plt.figure(figsize=(24, 12), dpi=200)

    for i, (mask, threshold, e) in enumerate(neighborhoods):
        plt.subplot(2, 4, i+1)
        sns.heatmap(mask, cmap='Oranges', linewidths=1, linecolor="#cbcbcb", cbar=False)
        plt.title(f"Inferred Neighborhood\nThreshold: {threshold:.2g}, Energy: {e:.2g}", {'fontsize': 15})
        plt.xticks([]); plt.yticks([]);
        
        if i == 0:
            plt.ylabel(f"{model.__class__.__name__}\n{rule_str.replace('_', '/')}", rotation=90, fontdict={'fontsize': 15})

    plt.tight_layout()
    
    if out_dir is not None:
        plt.savefig(f"{out_dir}/{model.__class__.__name__}--{rule_str}_neighborhoods.png")