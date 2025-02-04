import gc
import os
import pickle
import sys
import time
from scene import Scene, GaussianModel
from argparse import ArgumentParser
from arguments import ModelParamsMatching, PipelineParams
from plyfile import PlyData, PlyElement
import torch
import faiss
import fpsample
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN


from utils.system_utils import mkdir_p

class LODKNNMatchingInterpolation:
    def __init__(self, dataset1, dataset2, pipe, iteration1=None, iteration2=None, replace_2_from_1_part=False):
        """
        Initialize two Gaussian models from different datasets
        Args:
            dataset1: First dataset
            dataset2: Second dataset
            pipe: Pipeline parameters
            iteration1: Iteration number for first model
            iteration2: Iteration number for second model
        """
        # Initialize first Gaussian model and scene
        self.gaussians1 = GaussianModel(dataset1.sh_degree)
        self.scene1 = Scene(dataset1, self.gaussians1, load_iteration=iteration1, shuffle=False)
        
        # Initialize second Gaussian model and scene
        self.gaussians2 = GaussianModel(dataset2.sh_degree)
        self.scene2 = Scene(dataset2, self.gaussians2, load_iteration=iteration2, shuffle=False)

        # print number of points in gaussians1 and gaussians2
        print(f"Number of points in gaussians1: {self.gaussians1._xyz.shape[0]}")
        print(f"Number of points in gaussians2: {self.gaussians2._xyz.shape[0]}")
        
        # Store pipeline parameters
        self.pipe = pipe

        if replace_2_from_1_part:
            max_x_1, max_y_1, max_z_1 = self.gaussians1.get_xyz.max(dim=0)[0]
            min_x_1, min_y_1, min_z_1 = self.gaussians1.get_xyz.min(dim=0)[0]
            min_x_2, min_y_2, min_z_2 = self.gaussians2.get_xyz.min(dim=0)[0]
            max_x_2, max_y_2, max_z_2 = self.gaussians2.get_xyz.max(dim=0)[0]

            # Define the range for replacement
            threshold_x_1 = min_x_1 + 0.6 * (max_x_1 - min_x_1)
            # threshold_y_1 = min_y_1 + 0.05 * (max_y_1 - min_y_1)
            # threshold_z_1 = min_z_1 + 0.05 * (max_z_1 - min_z_1   )
            threshold_x_2 = min_x_2 + 1 * (max_x_2 - min_x_2)
            # threshold_y_2 = min_y_2 + 0.05 * (max_y_2 - min_y_2)
            # threshold_z_2 = min_z_2 + 0.05 * (max_z_2 - min_z_2)

            # Create a mask for points in gaussians1 that fall within the specified range
            mask1 = (self.gaussians1.get_xyz[:, 0] >= min_x_1) & (self.gaussians1.get_xyz[:, 0] <= threshold_x_1) \
                # & \
                #      (self.gaussians1.get_xyz[:, 1] >= min_y) & (self.gaussians1.get_xyz[:, 1] <= threshold_y) & \
                #      (self.gaussians1.get_xyz[:, 2] >= min_z) & (self.gaussians1.get_xyz[:, 2] <= threshold_z)

            # Create a mask for points in gaussians2 that fall within the specified range
            mask2 = (self.gaussians2.get_xyz[:, 0] >= min_x_2) & (self.gaussians2.get_xyz[:, 0] <= threshold_x_1) \
                # & \
                #      (self.gaussians2.get_xyz[:, 1] >= min_y) & (self.gaussians2.get_xyz[:, 1] <= threshold_y) & \
                #      (self.gaussians2.get_xyz[:, 2] >= min_z) & (self.gaussians2.get_xyz[:, 2] <= threshold_z)

            print("Number of points in gaussians1 before removing points: ", self.gaussians1._xyz.shape[0])
            print("Number of points in gaussians2 before removing points: ", self.gaussians2._xyz.shape[0])
            # Remove points in gaussians1 that are masked by mask1
            self.gaussians1._xyz = self.gaussians1._xyz[~mask1]
            self.gaussians1._features_dc = self.gaussians1._features_dc[~mask1]
            self.gaussians1._features_rest = self.gaussians1._features_rest[~mask1]
            self.gaussians1._opacity = self.gaussians1._opacity[~mask1]
            self.gaussians1._scaling = self.gaussians1._scaling[~mask1]
            self.gaussians1._rotation = self.gaussians1._rotation[~mask1]
            self.gaussians1._semantic_feature = self.gaussians1._semantic_feature[~mask1]
            print("Number of points in gaussians1 after removing points: ", self.gaussians1._xyz.shape[0])

            part_gaussians2_xyz = self.gaussians2._xyz[mask2]
            part_gaussians2_features_dc = self.gaussians2._features_dc[mask2]
            part_gaussians2_features_rest = self.gaussians2._features_rest[mask2]
            part_gaussians2_opacity = self.gaussians2._opacity[mask2]
            part_gaussians2_scaling = self.gaussians2._scaling[mask2]
            part_gaussians2_rotation = self.gaussians2._rotation[mask2]
            part_gaussians2_semantic_feature = self.gaussians2._semantic_feature[mask2]

            print("Number of points in gaussians2 after removing points: ", part_gaussians2_xyz.shape[0])

            # Extend gaussians1 with points from gaussians2 that are masked by mask2
            self.gaussians1._xyz = torch.cat((self.gaussians1._xyz, part_gaussians2_xyz), dim=0)
            self.gaussians1._features_dc = torch.cat((self.gaussians1._features_dc, part_gaussians2_features_dc), dim=0)
            self.gaussians1._features_rest = torch.cat((self.gaussians1._features_rest, part_gaussians2_features_rest), dim=0)
            self.gaussians1._opacity = torch.cat((self.gaussians1._opacity, part_gaussians2_opacity), dim=0)
            self.gaussians1._scaling = torch.cat((self.gaussians1._scaling, part_gaussians2_scaling), dim=0)
            self.gaussians1._rotation = torch.cat((self.gaussians1._rotation, part_gaussians2_rotation), dim=0)
            self.gaussians1._semantic_feature = torch.cat((self.gaussians1._semantic_feature, part_gaussians2_semantic_feature), dim=0)

            print("Number of points in gaussians1 after adding points: ", self.gaussians1._xyz.shape[0])

    @staticmethod
    def parse_args():
        parser = ArgumentParser(description="KNN Matching parameters")
        
        # Create model parameter instances with postfixes
        model_params1 = ModelParamsMatching(parser, postfix="_1")
        model_params2 = ModelParamsMatching(parser, postfix="_2")
        
        pipe_params = PipelineParams(parser)
        
        # Add iterations for both models
        parser.add_argument('--iteration_1', type=int, default=7000)
        parser.add_argument('--iteration_2', type=int, default=7000)

        # Add group_pkl_paths
        parser.add_argument('--group_pkl_paths', nargs='+', type=str, default=None)

        # Add matching_features
        parser.add_argument('--matching_features', nargs='+', type=str, default=['semantic_feature', 'dir_centers'], choices=['semantic_feature', 'xyz', 'sr_features', 'dir_centers', 'sdf'])

        # Add interpolation_features
        parser.add_argument('--interpolation_features', nargs='+', type=str, default=['xyz', 'features_dc', 'features_rest', 'scaling', 'opacity', 'rotation'], choices=['xyz', 'features_dc', 'features_rest', 'scaling', 'opacity', 'rotation', 'semantic_feature'])

        args = parser.parse_args(sys.argv[1:])
        return args, model_params1, model_params2, pipe_params

    def load_gaussians(self, dataset1, dataset2, pipe, iteration1, iteration2):
        '''
        Load gaussians from two datasets
        '''
        self.gaussians1 = GaussianModel(dataset1.sh_degree)
        self.scene1 = Scene(dataset1, self.gaussians1, load_iteration=iteration1, shuffle=False)
        self.gaussians2 = GaussianModel(dataset2.sh_degree)
        self.scene2 = Scene(dataset2, self.gaussians2, load_iteration=iteration2, shuffle=False)
        self.pipe = pipe

    def release_gaussians(self):
        del self.gaussians1
        del self.gaussians2
        del self.scene1
        del self.scene2
        gc.collect()
        torch.cuda.empty_cache()
    def construct_list_of_attributes(self, gaussians):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(gaussians['features_dc'].shape[1]*gaussians['features_dc'].shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(gaussians['features_rest'].shape[1]*gaussians['features_rest'].shape[2]):
            l.append('f_rest_{}'.format(i))

        l.append('opacity')
        for i in range(gaussians['scaling'].shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(gaussians['rotation'].shape[1]):
            l.append('rot_{}'.format(i))
        # Add semantic features
        if 'semantic_feature' in gaussians:
            for i in range(gaussians['semantic_feature'].shape[1]*gaussians['semantic_feature'].shape[2]):  
                l.append('semantic_{}'.format(i))
        return l

    def save_new_gaussians(self, gaussians, path_gaussian, is_interpolation=False):
        mkdir_p(os.path.dirname(path_gaussian))
        xyz = gaussians['xyz'].detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = gaussians['features_dc'].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = gaussians['features_rest'].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = gaussians['opacity'].detach().cpu().numpy()
        scale = gaussians['scaling'].detach().cpu().numpy()
        rotation = gaussians['rotation'].detach().cpu().numpy()
        if not is_interpolation:
            semantic_feature = gaussians['semantic_feature'].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes(gaussians)]
        if is_interpolation:
            dtype_full = [e for e in dtype_full if 'semantic' not in e[0]]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        if not is_interpolation:
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, semantic_feature), axis=1) 
        else:
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1) 
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path_gaussian)
    
def furtherest_sampling(vecs, num_points):
        '''
        Sample points from vecs based on the furthest point sampling
        Args:
            vecs: (N, D) vectors to sample
            num_points: Number of points to sample
        Returns:
            pos: Sampled points dim: (num_points, D)
            indices: Indices of the sampled points dim: (num_points, 1)
        '''

        kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(vecs, num_points, h=3)

        sampled_points = vecs[kdline_fps_samples_idx]
        return sampled_points, kdline_fps_samples_idx


def group_points_by_knn_with_semantics(
    gaussians_vecs,  # shape: (N, D_vec (3+D))
    center_vecs,  # shape: (M, 3+D)
    xyz_weight=1.0,
    semantic_weight=1.0,
):
    """
    Assign each point in `xyz` to exactly one furthest-sampled center using KNN 
    (for candidate centers) and tie-break in 3D distance by using the semantic feature.

    Args:
        gaussians_vecs (np.ndarray): (N, D_vec (3+D)) gaussians vectors
        center_vecs (np.ndarray):      (M, D_vec (3+D)) weighted center vectors (xyz + semantic features) for each center.
        xyz_weight (float):           Weight for xyz in center vectors.
        semantic_weight (float):      Weight for semantic features in center vectors.

    Returns:
        assignment (np.ndarray): shape (N,). assignment[p] = index c_i in [0..M-1].
    """
    N = gaussians_vecs.shape[0]
    M = center_vecs.shape[0]
    D_vec = gaussians_vecs.shape[1]


    # Use FAISS for fast KNN search
    d = D_vec  # dimensionality of points
    # Create a FAISS index on the GPU
    res = faiss.StandardGpuResources()  # create GPU resources
    index = faiss.IndexFlatL2(d)  # using L2 distance
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)  # transfer index to GPU
    
    # Add points to the index
    gpu_index.add(center_vecs.astype(np.float32))  # FAISS requires float32
    
    # Search for 1 nearest neighbor for each point
    # Returns distances and indices matrices of shape (N, 1)
    dist_matrix, idx_matrix = gpu_index.search(gaussians_vecs.astype(np.float32), k=1)

    # Prepare assignment array
    assignment = np.full((N,), fill_value=-1, dtype=int)

    # Assign each point to the nearest center
    assignment = idx_matrix.flatten()

    return assignment

def reassign_centers(gaussians_vecs, num_centers, assignment, original_centers, xyz_weight=1.0, semantic_weight=1.0):
    '''
    Reassign centers using the assignment
    Args:
        gaussians_vecs: (N, D_vec (3+D)) gaussians vectors
        num_centers: number of centers
        assignment: (N, 1) assignment of each point to one center
        original_centers: (M, D_vec (3+D)) original centers
        xyz_weight: weight for xyz
        semantic_weight: weight for semantic features
    Returns:
        center_vecs: (num_centers, D_vec (3+D)) centers
    '''
    D_vec = gaussians_vecs.shape[1]
    center_vecs = np.zeros((num_centers, D_vec))
    for i in range(num_centers):
        # if len(gaussians_vecs[assignment == i]) > 0:
        center_vecs[i] = np.mean(gaussians_vecs[assignment == i], axis=0)
        # else:
        #     print(f"Warning: {i} has no points assigned to it, using original center")
        #     center_vecs[i] = original_centers[i]
    return center_vecs

def sample_new_center(gaussians_vecs, center_vecs, assignment, xyz_weight=1.0, semantic_weight=1.0):
    '''
    Sample a new center using furthest point sampling
    args:
        gaussians_vecs: (N, D_vec (3+D)) gaussians vectors
        center_vecs: (M, D_vec (3+D)) centers
        xyz_weight: weight for xyz
        semantic_weight: weight for semantic features
        assignment: (N,) assignment of each point to one center
    returns:
        center_vecs: (M+1, D_vec (3+D)) centers
        assignment: (N,) new assignment of each point to one center
    '''
    M = center_vecs.shape[0]
    N = gaussians_vecs.shape[0]
    D_vec = gaussians_vecs.shape[1]
    # assign the last center using furthest point sampling
    furthest_avg_dist = 0
    for i in range(N):
        vec = gaussians_vecs[i]
        dist_vec = np.linalg.norm(vec - center_vecs, axis=1)
        cur_avg_dist = np.mean(dist_vec)
        if cur_avg_dist > furthest_avg_dist:
            # check if the original center for this point has only one point assigned to it. Otherwise, after reassigb this point to a new center, the original center will have no points assigned to it.
            center_idx = assignment[i]
            if np.sum(assignment == center_idx) != 1:
                furthest_avg_dist = cur_avg_dist
                furthest_vec_idx = i
    # extend center_vecs' size by 1
    center_vecs = np.concatenate((center_vecs, np.zeros((1, D_vec))), axis=0)
    center_vecs[-1] = gaussians_vecs[furthest_vec_idx]
    assignment[furthest_vec_idx] = M
    return center_vecs, assignment

def your_fps_and_knn_grouping(gaussians, num_centers, xyz_weight=1.0, semantic_weight=1.0):
    """
    1) Furthest point sample m centers (total number of centers=num_centers).
    2) Reassign centers using KNN assignment.
    3) Sample a new center using furthest point sampling.
    4) Repeat 2) and 3) until the number of centers reaches num_centers. (Before 3), make sure for each number of centers, the assignment converges)

    Args:
        gaussians: some GaussianModel with .get_xyz, .get_semantic_feature
        num_centers (int): number of furthest-sampled centers
        xyz_weight (float): weight for xyz
        semantic_weight (float): weight for semantic features

    Returns:
        centers_xyz (np.ndarray): shape (M, 3), the 3D coords of chosen centers
        groups (list of lists): groups[i] = list of point indices assigned to center i
        assignment (np.ndarray): shape (N, 1), assignment of each point to one center
    """
    xyz = gaussians.get_xyz.detach().cpu().numpy()       # shape (N, 3)
    semantic_feats = gaussians.get_semantic_feature.detach().cpu().numpy()  
    semantic_feats = semantic_feats.squeeze(1)
    # -> shape (N, D).  Make sure it matches (N, D). We have an extra dimension,
    gaussians_vecs = np.concatenate((xyz_weight * xyz, semantic_weight * semantic_feats), axis=1)


    cur_num_centers = 2

    # Step 1: Furthest Point Sampling 2 centers
    centers_xyz, centers_idx = furtherest_sampling(xyz, cur_num_centers)
    cur_center_vecs = np.concatenate((xyz_weight * xyz[centers_idx], semantic_weight * semantic_feats[centers_idx]), axis=1)
    assignment = group_points_by_knn_with_semantics(gaussians_vecs, cur_center_vecs, xyz_weight, semantic_weight)

    # Step 2: iteratively reassign centers, sampling new centers, and KNN assignment
    while cur_num_centers < num_centers:
        print(f"Current number of centers: {cur_num_centers}")
        # while assignment does not converge, reassign centers
        while True:
            assignment_prev = assignment
            cur_center_vecs = reassign_centers(gaussians_vecs, cur_num_centers, assignment, cur_center_vecs, xyz_weight, semantic_weight)
            assignment = group_points_by_knn_with_semantics(gaussians_vecs, cur_center_vecs, xyz_weight, semantic_weight)
            if np.all(assignment == assignment_prev):
                break
        cur_center_vecs, assignment = sample_new_center(gaussians_vecs, cur_center_vecs, assignment, xyz_weight, semantic_weight)
        cur_num_centers += 1

    while True:
        assignment_prev = assignment
        cur_center_vecs = reassign_centers(gaussians_vecs, cur_num_centers, assignment, cur_center_vecs, xyz_weight, semantic_weight)
        assignment = group_points_by_knn_with_semantics(gaussians_vecs, cur_center_vecs, xyz_weight, semantic_weight)
        if np.all(assignment == assignment_prev):
            break

    # Build final groups
    N = xyz.shape[0]
    M = cur_num_centers
    groups = [[] for _ in range(M)]
    for p in range(N):
        c_i = assignment[p]
        groups[c_i].append(p)
    centers_xyz = cur_center_vecs[:, :3]/xyz_weight
    # # print average distance between points in each group
    # for i in range(M):
    #     group = groups[i]
    #     if len(group) > 0:
    #         avg_dist = np.mean(np.linalg.norm(xyz[group] - centers_xyz[i], axis=1))
    #         print(f"Average distance between points in group {i}: {avg_dist}")
    return centers_xyz, groups, assignment


def write_points_to_ply(points, filename="output.ply"):
    """
    Write N x 3 points into a .ply file for visualization in MeshLab.

    Parameters:
        points (numpy.ndarray): A numpy array of shape (N, 3) representing points.
        filename (str): The name of the output .ply file.
    """
    if points.shape[1] != 3:
        raise ValueError("Input points must have shape (N, 3).")
    
    num_points = points.shape[0]
    with open(filename, "w") as file:
        # Write the PLY header
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {num_points}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("end_header\n")
        
        # Write the points
        for point in points:
            file.write(f"{point[0]} {point[1]} {point[2]}\n")

    print(f"File saved as {filename}")


def geometric_median(points, tol=1e-5, max_iter=200):
    """
    Compute the geometric median of a set of points using Weiszfeld's algorithm.
    
    Parameters:
        points (np.ndarray): Array of shape (n_points, k) representing the points.
        tol (float): Convergence tolerance.
        max_iter (int): Maximum number of iterations.
        
    Returns:
        np.ndarray: The geometric median of the input points (shape: (k,)).
    """
    # Initialize with the arithmetic mean
    median = np.mean(points, axis=0)
    
    for _ in range(max_iter):
        # Compute Euclidean distances from the current median to each point.
        distances = np.linalg.norm(points - median, axis=1)
        
        # Check for any distances that are effectively zero to avoid division by zero.
        nonzero = distances > tol
        if not np.any(nonzero):
            # All points are effectively at the current median.
            return median
        
        # Compute weights: 1/distance for non-zero distances.
        weights = np.zeros_like(distances)
        weights[nonzero] = 1.0 / distances[nonzero]
        
        # Update the median estimate.
        numerator = np.sum(points[nonzero] * weights[nonzero, None], axis=0)
        denominator = np.sum(weights[nonzero])
        new_median = numerator / denominator
        
        # Check for convergence.
        if np.linalg.norm(new_median - median) < tol:
            return new_median
        
        median = new_median

    return median  # Return the result if max_iter is reached

def robust_semantic_aggregation(X, eps=0.3, min_samples=10, tol=1e-5, max_iter=200):
    """
    Aggregate a set of semantic feature vectors robustly by:
      1. L2-normalizing each vector.
      2. Running DBSCAN to remove outliers.
      3. Selecting the largest (most dense) cluster.
      4. Computing the geometric median of that cluster.
    
    Parameters:
        X (np.ndarray): Input array of shape (n_points, k), where each row is a semantic feature vector.
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other (DBSCAN parameter).
        min_samples (int): The minimum number of samples in a neighborhood to form a cluster (DBSCAN parameter).
        tol (float): Tolerance for convergence in the geometric median computation.
        max_iter (int): Maximum iterations for the geometric median computation.
        
    Returns:
        np.ndarray: A representative semantic feature vector of shape (k,).
    """
    # Step 1: L2-normalization of each feature vector.
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    # Prevent division by zero (if any vector is all zeros, leave it as zeros).
    norms[norms == 0] = 1
    X_normalized = X / norms

    # Step 2: Run DBSCAN to cluster the normalized vectors.
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_normalized)
    
    # Step 3: Select the largest cluster (ignoring noise points labeled as -1).
    # If all points are labeled as noise, then use all points.
    valid_labels = labels[labels != -1]
    if valid_labels.size == 0:
        cluster_points = X_normalized
    else:
        unique_labels, counts = np.unique(valid_labels, return_counts=True)
        largest_cluster_label = unique_labels[np.argmax(counts)]
        cluster_points = X_normalized[labels == largest_cluster_label]
        # print the number of clusters
        print(f"Number of clusters: {len(unique_labels)}")
        # print the number of points in the largest cluster
        print(f"Number of points in the largest cluster: {counts[np.argmax(counts)]}")
    
    # Step 4: Compute the geometric median of the selected cluster.
    representative_vector = geometric_median(cluster_points, tol=tol, max_iter=max_iter)
    
    return representative_vector


def compute_group_features(gaussians, groups):
    """
    Compute average semantic features for each group of points
    Args:
        gaussians: GaussianModel
        groups: List of lists containing point indices for each group
    Returns:
        group_features: Average semantic features for each group (M, D)
    """
    semantic_features = gaussians.get_semantic_feature.detach().cpu().numpy()
    semantic_features = semantic_features.squeeze(1)
    group_features = []
    
    for group in groups:
        group_semantic = semantic_features[group]
        avg_semantic = robust_semantic_aggregation(group_semantic)
        # avg_semantic = np.mean(group_semantic, axis=0)
        group_features.append(avg_semantic)
    #     # print average of the entry in semantic_features
    #     print(f"Average of the entry in semantic_features: {np.mean(semantic_features)}")
    # # print average distance between points in each group
    # for i in range(len(group_features)):
    #     group = group_features[i].reshape(1, -1)
    #     if len(group) > 0:
    #         avg_dist = np.mean(np.linalg.norm(group_semantic - group, axis=1))
    #         print(f"Average distance between points in group in terms of features {i}: {avg_dist}")
    group_features = np.array(group_features)
    # # print shape of group_features
    # print(f"Shape of group_features: {group_features.shape}")
    return group_features


def match_groups_and_points(gaussians1, gaussians2, num_centers, xyz_weight, semantic_weight, method, group_pkl_paths):
    """
    Perform hierarchical matching:
    1. Match groups using average semantic features with Hungarian algorithm for one-to-one mapping
    2. Match points within matched groups using bidirectional KNN
    """
    # check if group_pkl_paths is a list of two paths (strings)
    if not isinstance(group_pkl_paths, list) or len(group_pkl_paths) != 2:
        raise ValueError("group_pkl_paths must be a list of two paths")
    group_pkl_path_1 = group_pkl_paths[0]
    group_pkl_path_2 = group_pkl_paths[1]
    # check if the two paths exist
    if not os.path.exists(group_pkl_path_1) or not os.path.exists(group_pkl_path_2):
        # First get groups for both gaussian sets
        centers_xyz1, groups1, assignment1 = your_fps_and_knn_grouping(
            gaussians1, num_centers, xyz_weight=xyz_weight, semantic_weight=semantic_weight
        )
        centers_xyz2, groups2, assignment2 = your_fps_and_knn_grouping(
            gaussians2, num_centers, xyz_weight=xyz_weight, semantic_weight=semantic_weight
        )
        # save centers_xyz1 and centers_xyz2 to ply files
        write_points_to_ply(centers_xyz1, "furthest_sampled_gaussians1.ply")
        write_points_to_ply(centers_xyz2, "furthest_sampled_gaussians2.ply")

        # save groups1 and groups2 which are list of lists
        with open(group_pkl_path_1, "wb") as f:
            pickle.dump(groups1, f)
        with open(group_pkl_path_2, "wb") as f:
            pickle.dump(groups2, f)
    else:
        # load groups1 and groups2
        with open(group_pkl_path_1, "rb") as f:
            groups1 = pickle.load(f)
        with open(group_pkl_path_2, "rb") as f:
            groups2 = pickle.load(f)
    
    
    # Compute average semantic features for each group
    group_features1 = compute_group_features(gaussians1, groups1)
    group_features2 = compute_group_features(gaussians2, groups2)

    # # print shape of group_features1 and group_features2
    # print(f"Shape of group_features1: {group_features1.shape}")
    # print(f"Shape of group_features2: {group_features2.shape}")
    
    # Option 1: Use Hungarian algorithm to find optimal one-to-one matching
    if method == "hungarian":
        # Compute pairwise distances between all groups
        cost_matrix = np.zeros((len(groups1), len(groups2)))
        for i in range(len(groups1)):
            for j in range(len(groups2)):
                cost_matrix[i,j] = np.sum((group_features1[i] - group_features2[j]) ** 2)
        # # print shape of cost_matrix
        # print(f"Shape of cost_matrix: {cost_matrix.shape}")
        # Option 1: Use Hungarian algorithm to find optimal one-to-one matching
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Initialize containers for final matches
        all_matches_1to2 = []
        all_matches_2to1 = []
        
        # For each matched group pair, perform bidirectional KNN matching
        for i, j in zip(row_ind, col_ind):
            points1 = groups1[i]
            points2 = groups2[j]
            
            # # print shape of points1 and points2
            # print(f"Shape of points1: {len(points1)}")
            # print(f"Shape of points2: {len(points2)}")
            
            if len(points1) == 0 or len(points2) == 0:
                continue
                
            # Get features for points in these groups
            feat1 = gaussians1.get_semantic_feature[points1].detach().cpu().numpy()
            feat2 = gaussians2.get_semantic_feature[points2].detach().cpu().numpy()
            feat1 = feat1.squeeze(1)
            feat2 = feat2.squeeze(1)
            
            # Create faiss indices for this group pair
            d = feat1.shape[1]
            index1 = faiss.IndexFlatL2(d)
            index2 = faiss.IndexFlatL2(d)
            
            index1.add(feat1)
            index2.add(feat2)
            
            # Compute bidirectional matches within groups
            _, matches_2to1 = index1.search(feat2, k=1)  # (N2, 1)
            _, matches_1to2 = index2.search(feat1, k=1)  # (N1, 1)
            
            # Convert local indices to global indices
            for local_idx1, local_idx2 in enumerate(matches_1to2):
                global_idx1 = points1[local_idx1]
                global_idx2 = points2[local_idx2[0]]
                all_matches_1to2.append((global_idx1, global_idx2))
                
            for local_idx2, local_idx1 in enumerate(matches_2to1):
                global_idx2 = points2[local_idx2]
                global_idx1 = points1[local_idx1[0]]
                all_matches_2to1.append((global_idx2, global_idx1))
    # Option 2: Use 1-nearest neighbor to find optimal one-to-one bidirectional matching
    elif method == "knn":
        # Create FAISS indices for both sets of group features
        d = group_features1.shape[1]  # dimensionality of features
        res = faiss.StandardGpuResources()  # create GPU resources
        
        # Create indices for both directions
        index1 = faiss.IndexFlatL2(d)
        index2 = faiss.IndexFlatL2(d)
        gpu_index1 = faiss.index_cpu_to_gpu(res, 0, index1)
        gpu_index2 = faiss.index_cpu_to_gpu(res, 0, index2)
        
        # Add features to indices
        gpu_index1.add(group_features1.astype(np.float32))
        gpu_index2.add(group_features2.astype(np.float32))
        
        # Find nearest neighbors in both directions
        _, group_matching_2to1 = gpu_index1.search(group_features2.astype(np.float32), k=1)  # N2 x 1
        _, group_matching_1to2 = gpu_index2.search(group_features1.astype(np.float32), k=1)  # N1 x 1

        # convert group_matching_2to1 and group_matching_1to2 to numpy array
        group_matching_2to1 = group_matching_2to1.astype(int).squeeze(1)
        group_matching_1to2 = group_matching_1to2.astype(int).squeeze(1)

        # Initialize containers for final matches
        all_matches_1to2 = []
        all_matches_2to1 = []

        # Compute matches1to2
        for pts1_idx, pts2_idx in enumerate(group_matching_1to2):
            points1 = groups1[pts1_idx]
            points2 = groups2[pts2_idx]

            # Get features for points in these groups
            feat1 = gaussians1.get_semantic_feature[points1].detach().cpu().numpy()
            feat2 = gaussians2.get_semantic_feature[points2].detach().cpu().numpy()
            feat1 = feat1.squeeze(1)
            feat2 = feat2.squeeze(1)

            # Create faiss indices for this group pair
            d = feat2.shape[1]
            index = faiss.IndexFlatL2(d)
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
            gpu_index.add(feat2)

            # Compute matches2to1
            _, matches_1to2 = gpu_index.search(feat1, k=1)
            # Convert local indices to global indices
            for local_idx1, local_idx2 in enumerate(matches_1to2):
                global_idx1 = points1[local_idx1]
                global_idx2 = points2[local_idx2[0]]
                all_matches_1to2.append((global_idx1, global_idx2))
        # Compute matches2to1
        for pts2_idx, pts1_idx in enumerate(group_matching_2to1):
            points2 = groups2[pts2_idx]
            points1 = groups1[pts1_idx]

            # Get features for points in these groups
            feat2 = gaussians2.get_semantic_feature[points2].detach().cpu().numpy()
            feat1 = gaussians1.get_semantic_feature[points1].detach().cpu().numpy()
            feat2 = feat2.squeeze(1)
            feat1 = feat1.squeeze(1)

            # Create faiss indices for this group pair
            d = feat1.shape[1]
            index = faiss.IndexFlatL2(d)
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
            gpu_index.add(feat1)

            # Compute matches1to2
            _, matches_2to1 = gpu_index.search(feat2, k=1)

            # Convert local indices to global indices
            for local_idx2, local_idx1 in enumerate(matches_2to1):
                global_idx2 = points2[local_idx2]
                global_idx1 = points1[local_idx1[0]]
                all_matches_2to1.append((global_idx2, global_idx1))
        
    # Convert to numpy arrays in format expected by create_matched_gaussians
    # Extract matches and indices for create_matched_gaussians
    matches_1to2_1idx, matches_1to2 = np.array(all_matches_1to2).T
    matches_2to1_1idx, matches_2to1 = np.array(all_matches_2to1).T

    # Sort and adjust the order
    order_1to2 = np.argsort(matches_1to2_1idx)
    order_2to1 = np.argsort(matches_2to1_1idx)

    matches_1to2 = matches_1to2[order_1to2].reshape(-1, 1)
    matches_2to1 = matches_2to1[order_2to1].reshape(-1, 1)
             
    # print length of all_matches_1to2 and all_matches_2to1, max and min of matched index
    print(f"Length of all_matches_1to2: {len(all_matches_1to2)}")
    print(f"Length of all_matches_2to1: {len(all_matches_2to1)}")
    print(f"Max of matched index in all_matches_1to2: {max([m[1] for m in all_matches_1to2])}")
    print(f"Min of matched index in all_matches_1to2: {min([m[1] for m in all_matches_1to2])}")
    print(f"Max of matched index in all_matches_2to1: {max([m[1] for m in all_matches_2to1])}")
    print(f"Min of matched index in all_matches_2to1: {min([m[1] for m in all_matches_2to1])}")
    # check if the first entry in tuple of all_matches_1to2 and all_matches_2to1 are unique
    if len(set([m[0] for m in all_matches_1to2])) != len(all_matches_1to2):
        raise ValueError("all_matches_1to2 is not unique")
    if len(set([m[0] for m in all_matches_2to1])) != len(all_matches_2to1):
        raise ValueError("all_matches_2to1 is not unique")
    
    # Create matched gaussians using existing function
    new_gaussians1, new_gaussians2, new_matches = create_matched_gaussians(gaussians1, gaussians2, matches_1to2, matches_2to1)
    return new_gaussians1, new_gaussians2, new_matches

def create_matched_gaussians(gaussians1, gaussians2, matches_1to2, matches_2to1):
    """
    Create matched gaussians with adjusted opacities based on number of links.
    
    Args:
        matches_1to2: Indices of matches from gaussians1 to gaussians2 (N1, 1)
        matches_2to1: Indices of matches from gaussians2 to gaussians1 (N2, 1)
        gaussians1: First GaussianModel
        gaussians2: Second GaussianModel
    
    Returns:
        new_gaussians1: Modified first GaussianModel with duplicated gaussians in dict format
        new_gaussians2: Modified second GaussianModel with duplicated gaussians in dict format
        new_matches: New matches between new_gaussians1 and new_gaussians2 format [(idx1, idx2), ...] shape (N1 + N2, 2)
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Use CPU for all operations except for indices
    k_values_1 = np.zeros(gaussians1.get_xyz.shape[0], dtype=np.int64)
    k_values_2 = np.zeros(gaussians2.get_xyz.shape[0], dtype=np.int64)
    
    # Count from matches_1to2
    for i in range(matches_1to2.shape[0]):
        k_values_1[i] += 1
        k_values_2[matches_1to2[i, 0]] += 1
        
    # Count from matches_2to1
    for i in range(matches_2to1.shape[0]):
        k_values_2[i] += 1
        k_values_1[matches_2to1[i, 0]] += 1
    # Create indices for duplicating gaussians
    indices_1 = []  # Indices into gaussians1
    indices_2 = []  # Indices into gaussians2
    new_matches = []  # Will store (idx1, idx2) pairs for final matching

    # k_values_1 and k_values_2 are the number of links for each gaussian
    # compute the cumulated sum of k_values_1 and k_values_2
    cum_k_values_1 = np.cumsum(k_values_1)
    cum_k_values_2 = np.cumsum(k_values_2)

    # current idx for matching for each gaussian
    current_idx_1 = np.zeros(gaussians1.get_xyz.shape[0], dtype=np.int64)
    current_idx_2 = np.zeros(gaussians2.get_xyz.shape[0], dtype=np.int64)
    current_idx_1[1:] = cum_k_values_1[:-1]
    current_idx_2[1:] = cum_k_values_2[:-1]

    # Process matches_1to2
    for i in range(matches_1to2.shape[0]):
        if k_values_1[i] > 0:  # If gaussian has any links
            # Add k copies of gaussian1[i]
            k = int(k_values_1[i])
            for _ in range(k):
                indices_1.append(i)
            
            # Add match to matches_1to2[i]
            j = matches_1to2[i, 0]
            new_matches.append((current_idx_1[i], current_idx_2[j]))
            current_idx_1[i] += 1
            current_idx_2[j] += 1
    
    # Process matches_2to1
    for i in range(matches_2to1.shape[0]):
        if k_values_2[i] > 0:  # If gaussian has any links
            # Add k copies of gaussian2[i]
            k = int(k_values_2[i])
            for _ in range(k):
                indices_2.append(i)
            
            # Add match to matches_2to1[i]
            j = matches_2to1[i, 0]
            new_matches.append((current_idx_1[j], current_idx_2[i]))
            current_idx_2[i] += 1
            current_idx_1[j] += 1
    
    # check if current_idx_1 and current_idx_2 are monotonic increasing
    if not all(current_idx_1[i] < current_idx_1[i+1] for i in range(len(current_idx_1)-1)):
        raise ValueError("current_idx_1 is not monotonic increasing")
    if not all(current_idx_2[i] < current_idx_2[i+1] for i in range(len(current_idx_2)-1)):
        raise ValueError("current_idx_2 is not monotonic increasing")
    
    # check if new_matches's elements in list contains first element of tuple and second element of tuple are unique
    first_indices = [m[0] for m in new_matches]
    second_indices = [m[1] for m in new_matches]
    len_first_indices = len(first_indices)
    len_unique_first_indices = len(set(first_indices))
    len_second_indices = len(second_indices)
    len_unique_second_indices = len(set(second_indices))

    if len_unique_first_indices != len_first_indices:
        raise ValueError("Duplicate first indices found in new_matches")

    if len_unique_second_indices != len_second_indices:
        raise ValueError("Duplicate second indices found in new_matches")

    # Move indices to GPU
    indices_1 = torch.tensor(indices_1, device=device)
    indices_2 = torch.tensor(indices_2, device=device)
    k_values_1 = torch.tensor(k_values_1, device=device)
    k_values_2 = torch.tensor(k_values_2, device=device)
    
    # Create new gaussians with duplicated points and adjusted opacities
    new_gaussians1 = {
        'xyz': gaussians1.get_xyz[indices_1],
        'features_dc': gaussians1._features_dc[indices_1],
        'features_rest': gaussians1._features_rest[indices_1],
        'scaling': gaussians1._scaling[indices_1],
        'rotation': gaussians1._rotation[indices_1],
        'opacity': gaussians1.inverse_opacity_activation(
            gaussians1.opacity_activation(gaussians1._opacity[indices_1]) / k_values_1[indices_1].reshape(-1, 1)
        ),
        'semantic_feature': gaussians1._semantic_feature[indices_1],
        'active_sh_degree': gaussians1.active_sh_degree
    }
    
    new_gaussians2 = {
        'xyz': gaussians2.get_xyz[indices_2],
        'features_dc': gaussians2._features_dc[indices_2],
        'features_rest': gaussians2._features_rest[indices_2],
        'scaling': gaussians2._scaling[indices_2],
        'rotation': gaussians2._rotation[indices_2],
        'opacity': gaussians2.inverse_opacity_activation(
            gaussians2.opacity_activation(gaussians2._opacity[indices_2]) / k_values_2[indices_2].reshape(-1, 1)
        ),
        'semantic_feature': gaussians2._semantic_feature[indices_2],
        'active_sh_degree': gaussians2.active_sh_degree
    }

    # Create reordered tensors based on new_matches
    reordered_indices1 = []
    reordered_indices2 = []
    for match in new_matches:
        reordered_indices1.append(match[0])
        reordered_indices2.append(match[1])
    
    # Convert to tensors and move to same device
    reordered_indices1 = torch.tensor(reordered_indices1, device=device)
    reordered_indices2 = torch.tensor(reordered_indices2, device=device)

    # Reorder all attributes of new_gaussians1 and new_gaussians2
    new_gaussians1 = {
        'xyz': new_gaussians1['xyz'][reordered_indices1],
        'features_dc': new_gaussians1['features_dc'][reordered_indices1],
        'features_rest': new_gaussians1['features_rest'][reordered_indices1],
        'scaling': new_gaussians1['scaling'][reordered_indices1],
        'rotation': new_gaussians1['rotation'][reordered_indices1],
        'opacity': new_gaussians1['opacity'][reordered_indices1],
        'semantic_feature': new_gaussians1['semantic_feature'][reordered_indices1],
        'active_sh_degree': new_gaussians1['active_sh_degree']
    }

    new_gaussians2 = {
        'xyz': new_gaussians2['xyz'][reordered_indices2],
        'features_dc': new_gaussians2['features_dc'][reordered_indices2],
        'features_rest': new_gaussians2['features_rest'][reordered_indices2],
        'scaling': new_gaussians2['scaling'][reordered_indices2],
        'rotation': new_gaussians2['rotation'][reordered_indices2],
        'opacity': new_gaussians2['opacity'][reordered_indices2],
        'semantic_feature': new_gaussians2['semantic_feature'][reordered_indices2],
        'active_sh_degree': new_gaussians2['active_sh_degree']
    }

    print("Number of points in new_gaussians1: ", new_gaussians1['xyz'].shape[0])
    print("Number of points in new_gaussians2: ", new_gaussians2['xyz'].shape[0])

    return new_gaussians1, new_gaussians2, new_matches

def linear_interpolate_gaussians(matcher, gaussians1, gaussians2, output_path, t=60, features_to_interpolate=[]):
        """
        Linearly interpolate between matched gaussians and save the sequence as PLY files.
        
        Args:
            output_path: Directory to save the interpolated PLY files
            t: Total number of frames to interpolate (including start and end)
        """
        mkdir_p(output_path)
        device = gaussians1['xyz'].device

        # Create interpolation ratios
        ratios = torch.linspace(0, 1, t, device=device)
        
        # Helper function for SLERP (Spherical Linear Interpolation)
        def slerp(q1, q2, t):
            dot = torch.sum(q1 * q2, dim=1, keepdim=True)
            # If dot product is negative, negate one of the quaternions to ensure shortest path
            q2 = torch.where(dot < 0, -q2, q2)
            dot = torch.where(dot < 0, -dot, dot)
            
            # If quaternions are very close, use linear interpolation
            DOT_THRESHOLD = 0.9995
            linear_interp = dot > DOT_THRESHOLD
            
            # Clamp dot product to valid range
            dot = torch.clamp(dot, -1.0, 1.0)
            omega = torch.acos(dot)
            sin_omega = torch.sin(omega)
            
            # Handle cases where sin_omega is close to zero
            mask = sin_omega.abs() < 1e-6
            s0 = torch.where(mask, 1.0 - t, torch.sin((1.0 - t) * omega) / sin_omega)
            s1 = torch.where(mask, t, torch.sin(t * omega) / sin_omega)
            
            # Linear interpolation for very close quaternions
            result = torch.where(
                linear_interp.expand(-1, 4),
                q1 * (1.0 - t) + q2 * t,
                s0 * q1 + s1 * q2
            )
            
            # Normalize the result
            return torch.nn.functional.normalize(result, dim=1)

        # For each frame
        for i, ratio in enumerate(ratios):
            print(f"Interpolating frame {i} with ratio {ratio}")
            interpolated = {}
            
            # Linear interpolation for most attributes
            for key in ['xyz', 'features_dc', 'features_rest', 'scaling', 'opacity']:
                if key in features_to_interpolate:
                    interpolated[key] = (1 - ratio) * gaussians1[key] + ratio * gaussians2[key]
                else:
                    interpolated[key] = gaussians1[key]
            # SLERP for rotation quaternions
            if 'rotation' in features_to_interpolate:
                interpolated['rotation'] = slerp(
                    gaussians1['rotation'],
                    gaussians2['rotation'],
                    ratio
                )
            else:
                interpolated['rotation'] = gaussians1['rotation']
            
            # Copy active_sh_degree
            interpolated['active_sh_degree'] = gaussians1['active_sh_degree']
            
            # Save the interpolated gaussians
            frame_path = os.path.join(output_path, f'frame_{i:04d}.ply')
            matcher.save_new_gaussians(interpolated, frame_path, is_interpolation=True)

def export_new_gaussians(matcher, new_gaussians1, new_gaussians2, path_gaussian1, path_gaussian2, is_interpolation=False):
    matcher.save_new_gaussians(new_gaussians1, path_gaussian1, is_interpolation)
    matcher.save_new_gaussians(new_gaussians2, path_gaussian2, is_interpolation)

if __name__ == "__main__":
    # Parse arguments
    args, model_params1, model_params2, pipe_params = LODKNNMatchingInterpolation.parse_args()
    
    # Extract parameters
    dataset1 = model_params1.extract(args)
    dataset2 = model_params2.extract(args)
    pipe = pipe_params.extract(args)

    matching_features = args.matching_features # list of features to use for matching [str]
    interpolation_features = args.interpolation_features # list of features to use for interpolation [str]

    # Create KNN matching instance
    print("Creating KNN matching instance. Loading gaussians...")
    matcher = LODKNNMatchingInterpolation(dataset1, dataset2, pipe, args.iteration_1, args.iteration_2, replace_2_from_1_part=False)

    
    print("Performing hierarchical matching...")
    N = matcher.gaussians1.get_xyz.shape[0]

    # # check the furthest sampled gaussians
    # pos1, _ = furtherest_sampling(matcher.gaussians1, num_points=2048)
    # pos2, _ = furtherest_sampling(matcher.gaussians2, num_points=2048)
    # write_points_to_ply(pos1, "furthest_sampled_gaussians1.ply")
    # write_points_to_ply(pos2, "furthest_sampled_gaussians2.ply")
    
    xyz_weight = 1.0/np.sqrt(3)
    semantic_weight = 1.0/np.sqrt(256)

    # print out the time of execution
    start_time = time.time()
    group_pkl_paths = args.group_pkl_paths
    new_gaussians1, new_gaussians2, new_matches = match_groups_and_points(matcher.gaussians1, matcher.gaussians2, num_centers=128, xyz_weight=xyz_weight, semantic_weight=semantic_weight, method="hungarian", group_pkl_paths=group_pkl_paths)
    end_time = time.time()
    print(f"Time of execution: {end_time - start_time} seconds")

    print("Saving new gaussians...")
    print("Exporting new gaussians...")
    path_new_gaussian1 = os.path.join(dataset1.model_path,
                                "point_cloud",
                                "iteration_" + str(args.iteration_1),
                                "new_point_cloud.ply")
    path_new_gaussian2 = os.path.join(dataset2.model_path,
                                "point_cloud",
                                "iteration_" + str(args.iteration_2),
                                "new_point_cloud.ply")
    export_new_gaussians(matcher, new_gaussians1, new_gaussians2, path_new_gaussian1, path_new_gaussian2)
    # print("Randomly sampling matched gaussians...")
    # matcher.random_sample_matched_gaussians(prec=0.01)
    print("Interpolating gaussians...")
    output_path_dir = os.path.join(dataset1.model_path, "point_cloud", "interpolation")
    linear_interpolate_gaussians(matcher, new_gaussians1, new_gaussians2, output_path_dir, t=120, features_to_interpolate=interpolation_features)
    print("End of the program")
