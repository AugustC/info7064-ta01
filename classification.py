from image_patch import ImagePatch


def classify_unlabeled_patches(patches_list : list[ImagePatch]):
    labeled_patches = [
        p for p in patches_list 
        if hasattr(p, 'patch_class') and p.patch_class is not None
    ]
    unlabeled_patches = [
        p for p in patches_list 
        if not hasattr(p, 'patch_class') or p.patch_class is None
    ]
    if not labeled_patches:
        raise ValueError("No labeled patches found. Please label at least one patch first.")
        
    print(f"References: {len(labeled_patches)} labeled patches.")
    print(f"Targets: {len(unlabeled_patches)} unlabeled patches. Classifying...\n")

    for i, target_patch in enumerate(unlabeled_patches):
        min_distance = float('inf')
        best_class = None
        
        for ref_patch in labeled_patches:
            dist = target_patch.calculate_distance(ref_patch)
            if dist < min_distance:
                min_distance = dist
                best_class = ref_patch.patch_class
                
        target_patch.patch_class = best_class
        if (i + 1) % 500 == 0 or (i + 1) == len(unlabeled_patches):
            print(f"Processed {i + 1}/{len(unlabeled_patches)} patches...")