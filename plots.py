import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches


def plot_patches(num_rows, num_cols, patch_samples, all_patches, top=5):
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(2.5 * num_cols, 2.5 * num_rows))
    if num_rows == 1:
        axes = [axes]
        
    for row_idx, original_patch in enumerate(patch_samples):
        ax_color = axes[row_idx][0]
        ax_color.imshow(original_patch.color_patch)
        ax_color.axis('off')
        ax_orig = axes[row_idx][1]
        ax_orig.imshow(original_patch.patch, cmap='gray')
        ax_orig.axis('off')
        if row_idx == 0:
            ax_orig.set_title("Original", fontweight='bold')
            
        similar_patches = original_patch.get_similar(all_patches, top=n_similar)
        for col_idx in range(n_similar):
            ax_sim = axes[row_idx][col_idx + 2]
            if col_idx < len(similar_patches):
                ax_sim.imshow(similar_patches[col_idx][0].patch, cmap='gray')
                if row_idx == 0:
                    ax_sim.set_title(f"Rank {col_idx + 2}")
            ax_sim.axis('off')
    
    plt.tight_layout()
    plt.savefig('patches_texture.png', bbox_inches='tight',dpi=300)
    plt.show()

def visualize_final_segmentation(patches_list, window_size, image_index=0, image_size=512, alpha=0.5):
    class_mapping = {'dog': 0, 'wall': 1, 'floor': 2, 'other': 3}
    colors = ['#d62728', '#1f77b4', '#2ca02c', '#7f7f7f'] 
    cmap = ListedColormap(colors)
    
    win_h, win_w = window_size
    grid_cols = image_size // win_w
    grid_rows = image_size // win_h
    patches_per_image = grid_rows * grid_cols
    
    start_idx = image_index * patches_per_image
    end_idx = start_idx + patches_per_image
    target_patches = patches_list[start_idx:end_idx]
    
    channels = target_patches[0].color_patch.shape[2] if len(target_patches[0].color_patch.shape) > 2 else 1
    full_image = np.zeros((image_size, image_size, channels), dtype=np.uint8)
    
    if channels == 1:
        full_image = full_image.squeeze()
    labels_1d = np.zeros(len(target_patches), dtype=int)
    
    for idx, patch in enumerate(target_patches):
        row = idx // grid_cols
        col = idx % grid_cols
        y_start, y_end = row * win_h, (row + 1) * win_h
        x_start, x_end = col * win_w, (col + 1) * win_w
        full_image[y_start:y_end, x_start:x_end] = patch.color_patch
        p_class = getattr(patch, 'patch_class', 'other')
        labels_1d[idx] = class_mapping.get(p_class, 3)

    segmentation_map = labels_1d.reshape((grid_rows, grid_cols))
    full_seg_map = np.repeat(np.repeat(segmentation_map, win_h, axis=0), win_w, axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    axes[0].imshow(full_image)
    axes[0].set_title(f"Original Image {image_index}", fontweight='bold')
    axes[0].axis('off')
    axes[1].imshow(full_image)
    axes[1].imshow(full_seg_map, cmap=cmap, alpha=alpha, interpolation='nearest', vmin=0, vmax=3)
    axes[1].set_title(f"Classified Segmentation {image_index}", fontweight='bold')
    axes[1].axis('off')
    legend_patches = [
        patches.Patch(color=colors[i], label=name.capitalize()) 
        for name, i in class_mapping.items()
    ]
    axes[1].legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    plt.tight_layout()
    plt.savefig('segmentation.png',bbox_inches='tight',dpi=300)
    plt.show()
