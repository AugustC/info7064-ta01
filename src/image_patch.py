import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from IPython.display import clear_output


class ImagePatch:

    def __init__(self, image, textures, x, y, height, width, color_img=None):
        self.patch = image[y:y+height, x:x+width]
        self.color_patch = color_img[y:y+height, x:x+width] if color_img is not None else None
        self.texture_vector = self.build_texture_vector(textures, x, y, height, width)
        self.patch_class = None

    def build_texture_vector(self, textures, x, y, height, width):
        vector = []
        for t in textures:
            patch = t[y:y+height, x:x+width]
            vector.append(np.mean(patch))
        return np.array(vector)

    def calculate_distance(self, patch):
        distance = np.linalg.norm(self.texture_vector - patch.texture_vector)
        return distance

    def get_similar(self, patches, top=5):
        patches_distances = []
        for p in patches:
            dist = self.calculate_distance(p)
            patches_distances.append((p, dist))
        sorted_patches = sorted(patches_distances, key=lambda x: x[1])
        return sorted_patches[1:top+1]


    
def get_patches(img, textures, window_size, color_img=None):
    img_h, img_w = img.shape
    win_h, win_w = window_size
    image_patches = []
    for y in range(0, img_h, win_h):
        for x in range(0, img_w, win_w):
            patch = ImagePatch(img, textures, x, y, win_h, win_w, color_img=color_img)
            image_patches.append(patch)

    return image_patches

def _get_diverse_indices(patches_list, k):
    num_patches = len(patches_list)
    if k >= num_patches:
        return list(range(num_patches))

    first_idx = random.randint(0, num_patches - 1)
    selected_indices = [first_idx]
    
    min_distances = np.array([
        patches_list[i].calculate_distance(patches_list[first_idx]) 
        for i in range(num_patches)
    ])
    
    for _ in range(1, k):
        farthest_idx = np.argmax(min_distances)
        selected_indices.append(farthest_idx)
        
        for i in range(num_patches):
            dist = patches_list[i].calculate_distance(patches_list[farthest_idx])
            if dist < min_distances[i]:
                min_distances[i] = dist
                
    return selected_indices

def label_diverse_patches_in_context(current_patches, k, window_size, full_image):
    win_h, win_w = window_size
    img_h, img_w = full_image.shape[:2]
    grid_cols = img_w // win_w
    
    valid_classes = {'1': 'dog', '2': 'wall', '3': 'floor', '4': 'other'}
    
    print(f"Calculating furthest points for {k} diverse regions...")
    
    sampled_indices = _get_diverse_indices(current_patches, k)
    clear_output(wait=True)
    
    for step, idx in enumerate(sampled_indices):
        patch_obj = current_patches[idx]
        row = idx // grid_cols
        col = idx % grid_cols
        x_start = col * win_w
        y_start = row * win_h
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(full_image)
        rect = patches.Rectangle(
            (x_start, y_start), win_w, win_h, 
            linewidth=3, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        
        ax.axis('off')
        ax.set_title(f"Labeling Region {step + 1} of {len(sampled_indices)}", fontweight='bold')
        
        plt.show(block=False)
        plt.pause(0.1)
        user_choice = None
        while user_choice not in valid_classes:
            print(f"\nClassify Region {step + 1}:")
            print("[1] Dog\n[2] Wall\n[3] Floor\n[4] Other")
            user_choice = input("Enter the number of the class (1-4): ").strip()
            if user_choice not in valid_classes:
                print("Invalid input. Please enter 1, 2, 3, or 4.")
                
        chosen_label = valid_classes[user_choice]
        patch_obj.patch_class = chosen_label
        plt.close(fig)
        clear_output(wait=True)