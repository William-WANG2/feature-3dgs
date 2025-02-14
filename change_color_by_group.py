import os
import torch
import random
import numpy as np
import colorsys
import torch.nn as nn

def generate_distinct_colors(n):
    """
    Generate n visually distinct colors using the golden ratio conjugate to space hues.
    Returns an array of RGB colors in the range [0, 1].
    """
    colors = []
    golden_ratio_conjugate = 0.618033988749895
    hue = random.random()  # Start at a random hue
    
    for _ in range(n):
        # Update hue using the golden ratio conjugate
        hue = (hue + golden_ratio_conjugate) % 1
        saturation = 0.9  # High saturation for vivid colors
        value = 0.9       # High brightness
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(rgb)
    
    return np.array(colors)

def color_gaussians_by_group(gaussians, groups):
    """
    Directly modify the colors of input gaussians based on their group assignments.
    
    Args:
        gaussians: GaussianModel instance
        groups: List of lists containing point indices for each group
    """
    # Generate distinct colors for each group
    n_groups = len(groups)
    colors = generate_distinct_colors(n_groups)
    
    # Create a color array for all points, initialized to black
    all_colors = np.zeros((gaussians.get_xyz.shape[0], 3))
    
    # Assign colors to points based on their group
    for group_idx, group in enumerate(groups):
        all_colors[group] = colors[group_idx]
    
    # Convert colors to SH coefficients (only DC term) and modify the gaussians
    # Create new tensors instead of modifying in-place
    new_features_dc = gaussians._features_dc.clone()
    new_features_dc[:, 0, 0] = torch.from_numpy(all_colors[:, 0]).float().to(gaussians._features_dc.device)
    new_features_dc[:, 0, 1] = torch.from_numpy(all_colors[:, 1]).float().to(gaussians._features_dc.device)
    new_features_dc[:, 0, 2] = torch.from_numpy(all_colors[:, 2]).float().to(gaussians._features_dc.device)
    
    # Assign the new features
    gaussians._features_dc = nn.Parameter(new_features_dc)
    
    # Set higher-order SH coefficients to zero
    gaussians._features_rest = nn.Parameter(torch.zeros_like(gaussians._features_rest))

if __name__ == "__main__":
    # Parse arguments similar to lod_knn_matching_interpolation.py
    from lod_knn_matching_interpolation import LODKNNMatchingInterpolation
    args, model_params1, model_params2, pipe_params = LODKNNMatchingInterpolation.parse_args()
    
    dataset1 = model_params1.extract(args)
    dataset2 = model_params2.extract(args)
    pipe = pipe_params.extract(args)
    
    # Create matcher instance
    matcher = LODKNNMatchingInterpolation(dataset1, dataset2, pipe, args.iteration_1, args.iteration_2)
    
    # Perform grouping
    xyz_weight = 4.0/np.sqrt(3)
    semantic_weight = 1.0/np.sqrt(256)
    
    # Get centers and groups
    _, groups1, _, _, groups2, _ = matcher.joint_fps_and_knn_grouping(
        matcher.gaussians1, matcher.gaussians2, 
        num_centers=16, 
        xyz_weight=xyz_weight, 
        semantic_weight=semantic_weight
    )
    
    # Color and save gaussians for both models
    output_path1 = os.path.join(dataset1.model_path, "point_cloud", "iteration_" + str(args.iteration_1), "colored_groups.ply")
    output_path2 = os.path.join(dataset2.model_path, "point_cloud", "iteration_" + str(args.iteration_2), "colored_groups.ply")
    
    color_gaussians_by_group(matcher.gaussians1, groups1, output_path1)
    color_gaussians_by_group(matcher.gaussians2, groups2, output_path2)

