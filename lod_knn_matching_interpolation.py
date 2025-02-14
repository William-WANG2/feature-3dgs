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
from scipy.spatial import cKDTree

from utils.system_utils import mkdir_p
from change_color_by_group import color_gaussians_by_group

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

        # Add group_matching_method
        parser.add_argument('--group_matching_method', type=str, default="hungarian", choices=["hungarian", "knn", "local_consistency_hungarian", "local_consistency_knn"])

        # Add matching_features
        parser.add_argument('--matching_features', nargs='+', type=str, default=['semantic_feature', 'dir_centers'], choices=['semantic_feature', 'xyz', 'sr_features', 'dir_centers', 'sdf'])

        # Add interpolation_features
        parser.add_argument('--interpolation_features', nargs='+', type=str, default=['xyz', 'features_dc', 'features_rest', 'scaling', 'opacity', 'rotation'], choices=['xyz', 'features_dc', 'features_rest', 'scaling', 'opacity', 'rotation', 'semantic_feature'])

        # Choose if we want to color the gaussians by group
        parser.add_argument('--color_by_group', action='store_true')

        # Choose if we want to use joint fps and knn grouping
        parser.add_argument('--use_joint_fps_and_knn_grouping', action='store_true')

        # Choose the number of centers for the grouping
        parser.add_argument('--num_centers', type=int, default=16)

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
    # USING MEAN OF CLUSTER
    # D_vec = gaussians_vecs.shape[1]
    # center_vecs = np.zeros((num_centers, D_vec))
    # for i in range(num_centers):
    #     # if len(gaussians_vecs[assignment == i]) > 0:
    #     center_vecs[i] = np.mean(gaussians_vecs[assignment == i], axis=0)
    #     # else:
    #     #     print(f"Warning: {i} has no points assigned to it, using original center")
    #     #     center_vecs[i] = original_centers[i]
    # return center_vecs

    # USING CLOSEST POINT TO MEAN OF CLUSTER
    D_vec = gaussians_vecs.shape[1]
    center_vecs = np.zeros((num_centers, D_vec))
    for i in range(num_centers):
        points_in_cluster = gaussians_vecs[assignment == i]
        if len(points_in_cluster) > 0:
            # Split into XYZ and remaining features
            xyz = points_in_cluster[:, :3]
            features = points_in_cluster[:, 3:]
            
            # Use median for XYZ coordinates
            center_vecs[i, :3] = np.median(xyz, axis=0)
            
            # Use mean for remaining features
            center_vecs[i, 3:] = np.mean(features, axis=0)
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

import numpy as np
import faiss

# ------------------------------
# Helper functions (unchanged)
# ------------------------------

# def furtherest_sampling(vecs, num_points):
#     """
#     Sample points from vecs based on furthest point sampling.
    
#     Args:
#         vecs (np.ndarray): (N, D) vectors.
#         num_points (int): Number of points to sample.
#     Returns:
#         sampled_points (np.ndarray): (num_points, D) sampled vectors.
#         indices (np.ndarray): indices of sampled points.
#     """
#     kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(vecs, num_points, h=3)
#     sampled_points = vecs[kdline_fps_samples_idx]
#     return sampled_points, kdline_fps_samples_idx


# def group_points_by_knn_with_semantics(
#     gaussians_vecs,  # shape: (N, D_vec = 3+D)
#     center_vecs,     # shape: (M, D_vec = 3+D)
#     xyz_weight=1.0,
#     semantic_weight=1.0,
# ):
#     """
#     Assign each point in gaussians_vecs to its nearest center in center_vecs.
#     The distance is computed in the weighted full-feature space.
    
#     Args:
#         gaussians_vecs (np.ndarray): (N, 3+D) array.
#         center_vecs (np.ndarray):   (M, 3+D) array.
#         xyz_weight (float): weight for xyz.
#         semantic_weight (float): weight for semantic features.
#     Returns:
#         assignment (np.ndarray): (N,) array; assignment[i] is the index of the nearest center.
#     """
#     N = gaussians_vecs.shape[0]
#     d = gaussians_vecs.shape[1]
#     res = faiss.StandardGpuResources()
#     index = faiss.IndexFlatL2(d)
#     gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
#     gpu_index.add(center_vecs.astype(np.float32))
#     _, idx_matrix = gpu_index.search(gaussians_vecs.astype(np.float32), k=1)
#     assignment = idx_matrix.flatten()
#     return assignment


# def reassign_centers(gaussians_vecs, num_centers, assignment, original_centers, xyz_weight=1.0, semantic_weight=1.0):
#     """
#     Recompute each center as the mean of the points assigned to it.
    
#     Args:
#         gaussians_vecs (np.ndarray): (N, 3+D) array.
#         num_centers (int): number of centers.
#         assignment (np.ndarray): (N,) current assignment of points.
#         original_centers (np.ndarray): (num_centers, 3+D) centers.
#     Returns:
#         center_vecs (np.ndarray): (num_centers, 3+D) updated centers.
#     """
#     D_vec = gaussians_vecs.shape[1]
#     center_vecs = np.zeros((num_centers, D_vec))
#     for i in range(num_centers):
#         pts = gaussians_vecs[assignment == i]
#         if pts.shape[0] > 0:
#             center_vecs[i] = np.mean(pts, axis=0)
#         else:
#             # If no points are assigned, fall back on the original center.
#             center_vecs[i] = original_centers[i]
#     return center_vecs


# def sample_new_center(gaussians_vecs, center_vecs, assignment, xyz_weight=1.0, semantic_weight=1.0):
#     """
#     Sample a new center for group A using full-feature FPS.
#     This function returns the updated centers, updated assignment, and the index
#     (in gaussians_vecs) of the newly selected center.
    
#     Args:
#         gaussians_vecs (np.ndarray): (N, 3+D) array.
#         center_vecs (np.ndarray): (M, 3+D) current centers.
#         assignment (np.ndarray): (N,) current assignment.
#     Returns:
#         new_center_vecs (np.ndarray): (M+1, 3+D) centers.
#         new_assignment (np.ndarray): (N,) updated assignment.
#         new_center_idx (int): index (in gaussians_vecs) of the new center.
#     """
#     M = center_vecs.shape[0]
#     N = gaussians_vecs.shape[0]
#     furthest_avg_dist = 0.0
#     new_center_idx = None
#     for i in range(N):
#         vec = gaussians_vecs[i]
#         dist_vec = np.linalg.norm(vec - center_vecs, axis=1)
#         cur_avg_dist = np.mean(dist_vec)
#         # Only choose this point if its current center has more than one point assigned.
#         if cur_avg_dist > furthest_avg_dist:
#             center_idx = assignment[i]
#             if np.sum(assignment == center_idx) != 1:
#                 furthest_avg_dist = cur_avg_dist
#                 new_center_idx = i

#     new_center_vec = gaussians_vecs[new_center_idx].reshape(1, -1)
#     new_center_vecs = np.concatenate((center_vecs, new_center_vec), axis=0)
#     assignment[new_center_idx] = M  # update assignment of the new center
#     return new_center_vecs, assignment, new_center_idx


# ------------------------------
# Joint grouping function with iterative refinement for both groups
# ------------------------------

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
    center_features = cur_center_vecs[:, 3:]/semantic_weight
    # # print average distance between points in each group
    # for i in range(M):
    #     group = groups[i]
    #     if len(group) > 0:
    #         avg_dist = np.mean(np.linalg.norm(xyz[group] - centers_xyz[i], axis=1))
    #         print(f"Average distance between points in group {i}: {avg_dist}")
    return centers_xyz, center_features, groups, assignment

def your_fps_and_knn_grouping_with_reference_centers(gaussians, num_centers, centers_features_reference, xyz_weight=1.0, semantic_weight=1.0):
    """
    Similar to your_fps_and_knn_grouping, but uses reference centers' features to guide initial center selection.
    
    Args:
        gaussians: GaussianModel with .get_xyz, .get_semantic_feature
        num_centers (int): number of centers to select
        xyz_weight (float): weight for xyz coordinates
        semantic_weight (float): weight for semantic features
        centers_features_reference (np.ndarray): shape (num_centers, D) reference center features
        
    Returns:
        centers_xyz (np.ndarray): shape (num_centers, 3) xyz coordinates of centers
        center_features (np.ndarray): shape (num_centers, D) semantic features of centers
        groups (list): list of lists containing point indices for each group
        assignment (np.ndarray): shape (N,) assignment of each point to a center
    """
    # Get point features and coordinates
    xyz = gaussians.get_xyz.detach().cpu().numpy()       # shape (N, 3)
    semantic_feats = gaussians.get_semantic_feature.detach().cpu().numpy()  
    semantic_feats = semantic_feats.squeeze(1)  # shape (N, D)
    
    # Combine features for distance computation
    gaussians_vecs = np.concatenate((xyz_weight * xyz, semantic_weight * semantic_feats), axis=1)

    
    # Step 1: Find initial centers by matching points to reference centers
    # Create FAISS index for semantic features
    d = semantic_feats.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(semantic_feats.astype(np.float32))
    
    # Find nearest points to reference centers
    _, initial_centers_idx = index.search(centers_features_reference.astype(np.float32), k=1)
    initial_centers_idx = initial_centers_idx.squeeze()
    
    # Get initial center vectors
    cur_center_vecs = np.concatenate(
        (xyz_weight * xyz[initial_centers_idx], 
         semantic_weight * semantic_feats[initial_centers_idx]), 
        axis=1
    )
    
    # Step 2: Initial assignment using KNN
    assignment = group_points_by_knn_with_semantics(
        gaussians_vecs, 
        cur_center_vecs, 
        xyz_weight, 
        semantic_weight
    )
    
    # Step 3: Iterative refinement until convergence
    while True:
        assignment_prev = assignment
        
        # Update centers based on current assignment
        cur_center_vecs = reassign_centers(
            gaussians_vecs, 
            num_centers, 
            assignment, 
            cur_center_vecs, 
            xyz_weight, 
            semantic_weight
        )
        
        # Reassign points to nearest centers
        assignment = group_points_by_knn_with_semantics(
            gaussians_vecs, 
            cur_center_vecs, 
            xyz_weight, 
            semantic_weight
        )
        
        # Check for convergence
        if np.all(assignment == assignment_prev):
            break
    
    # Build final groups
    N = xyz.shape[0]
    M = num_centers
    groups = [[] for _ in range(M)]
    for p in range(N):
        c_i = assignment[p]
        groups[c_i].append(p)
    
    # Extract final centers and features
    centers_xyz = cur_center_vecs[:, :3]/xyz_weight
    center_features = cur_center_vecs[:, 3:]/semantic_weight
    
    # Optional: Print average distances within groups
    for i in range(M):
        group = groups[i]
        if len(group) > 0:
            avg_dist = np.mean(np.linalg.norm(xyz[group] - centers_xyz[i], axis=1))
            print(f"Average distance between points in group {i}: {avg_dist}")
            
    return centers_xyz, center_features, groups, assignment

def joint_fps_and_knn_grouping(gaussians_A, gaussians_B, num_centers, xyz_weight=1.0, semantic_weight=1.0):
    """
    Joint grouping for two sets of gaussians.
    
    - For both groups, iterative assignment and center updates are performed
      in the full (xyz + semantic) space with the same weights.
    - However, when adding a new center for group B, the selection is driven solely by
      comparing the semantic feature of group B's points to the semantic feature of the
      newly selected center from group A.
    - In each iteration of the main loop, both groups undergo iterative refinement
      (until their assignments converge) before new centers are added.
    
    Args:
        gaussians_A: An object with attributes .get_xyz and .get_semantic_feature (group A).
        gaussians_B: An object with attributes .get_xyz and .get_semantic_feature (group B).
        num_centers (int): Desired total number of centers.
        xyz_weight (float): Weight for xyz (applied to both groups).
        semantic_weight (float): Weight for semantic features.
        
    Returns:
        centers_xyz_A (np.ndarray): (num_centers, 3) centers (xyz) for group A.
        groups_A (list of lists): groups_A[i] contains indices (into gaussians_A) assigned to center i.
        assignment_A (np.ndarray): (N_A,) assignment for group A.
        centers_xyz_B (np.ndarray): (num_centers, 3) centers (xyz) for group B.
        groups_B (list of lists): groups_B[i] contains indices (into gaussians_B) assigned to center i.
        assignment_B (np.ndarray): (N_B,) assignment for group B.
    """
    # ----- Prepare data for group A -----
    xyz_A = gaussians_A.get_xyz.detach().cpu().numpy()  # shape: (N_A, 3)
    semantic_A = gaussians_A.get_semantic_feature.detach().cpu().numpy().squeeze(1)  # shape: (N_A, D)
    # Build full feature vectors: [xyz_weight * xyz, semantic_weight * semantic]
    gaussians_A_vecs = np.concatenate((xyz_weight * xyz_A, semantic_weight * semantic_A), axis=1)
    
    # ----- Prepare data for group B (full representation) -----
    xyz_B = gaussians_B.get_xyz.detach().cpu().numpy()  # shape: (N_B, 3)
    semantic_B = gaussians_B.get_semantic_feature.detach().cpu().numpy().squeeze(1)  # shape: (N_B, D)
    gaussians_B_vecs = np.concatenate((xyz_weight * xyz_B, semantic_weight * semantic_B), axis=1)
    
    # ----- Initialization for Group A -----
    # Initialize centers via furthest sampling on xyz.
    centers_xyz_init, centers_idx_A = furtherest_sampling(xyz_A, 2)
    centers_A = np.concatenate(
        (xyz_weight * xyz_A[centers_idx_A], semantic_weight * semantic_A[centers_idx_A]),
        axis=1
    )
    assignment_A = group_points_by_knn_with_semantics(gaussians_A_vecs, centers_A, xyz_weight, semantic_weight)
    current_num_centers = centers_A.shape[0]  # initially 2
    
    # ----- Initialization for Group B -----
    # "Mirror" group A's centers: for each center in A, select a point in B whose semantic feature is closest.
    chosen_center_indices_B = set()  # to avoid duplicates
    centers_B_list = []
    for center in centers_A:
        target_sem = center[3:]  # semantic part of the center
        distances = np.linalg.norm(gaussians_B_vecs[:, 3:] - target_sem, axis=1)
        # Exclude points already chosen as centers.
        for idx in chosen_center_indices_B:
            distances[idx] = np.inf
        idx_B = np.argmin(distances)
        chosen_center_indices_B.add(idx_B)
        center_B_vec = gaussians_B_vecs[idx_B].reshape(1, -1)
        centers_B_list.append(center_B_vec)
    centers_B = np.concatenate(centers_B_list, axis=0)
    assignment_B = group_points_by_knn_with_semantics(gaussians_B_vecs, centers_B, xyz_weight, semantic_weight)
    
    # ----- Iterative Refinement and Center Addition -----
    while current_num_centers < num_centers:
        print("Current number of centers: ", current_num_centers)
        # Refine Group A until assignments converge.
        while True:
            prev_assignment_A = assignment_A.copy()
            centers_A = reassign_centers(gaussians_A_vecs, current_num_centers, assignment_A, centers_A, xyz_weight, semantic_weight)
            assignment_A = group_points_by_knn_with_semantics(gaussians_A_vecs, centers_A, xyz_weight, semantic_weight)
            if np.all(assignment_A == prev_assignment_A):
                break
        
        # Refine Group B until assignments converge.
        while True:
            prev_assignment_B = assignment_B.copy()
            centers_B = reassign_centers(gaussians_B_vecs, current_num_centers, assignment_B, centers_B, xyz_weight, semantic_weight)
            assignment_B = group_points_by_knn_with_semantics(gaussians_B_vecs, centers_B, xyz_weight, semantic_weight)
            if np.all(assignment_B == prev_assignment_B):
                break
        
        # --- Add new center for Group A ---
        centers_A, assignment_A, new_center_idx_A = sample_new_center(gaussians_A_vecs, centers_A, assignment_A, xyz_weight, semantic_weight)
        current_num_centers += 1
        
        # --- Add new center for Group B by semantic matching ---
        # Extract the semantic part of the new Group A center.
        new_center_sem = gaussians_A_vecs[new_center_idx_A][3:]
        distances = np.linalg.norm(gaussians_B_vecs[:, 3:] - new_center_sem, axis=1)
        # Exclude points already chosen as centers.
        for idx in chosen_center_indices_B:
            distances[idx] = np.inf
        new_center_idx_B = np.argmin(distances)
        chosen_center_indices_B.add(new_center_idx_B)
        new_center_B = gaussians_B_vecs[new_center_idx_B].reshape(1, -1)
        centers_B = np.concatenate((centers_B, new_center_B), axis=0)
        assignment_B = group_points_by_knn_with_semantics(gaussians_B_vecs, centers_B, xyz_weight, semantic_weight)
    
    # ----- Final Refinement (for both groups) -----
    while True:
        prev_assignment_A = assignment_A.copy()
        centers_A = reassign_centers(gaussians_A_vecs, current_num_centers, assignment_A, centers_A, xyz_weight, semantic_weight)
        assignment_A = group_points_by_knn_with_semantics(gaussians_A_vecs, centers_A, xyz_weight, semantic_weight)
        if np.all(assignment_A == prev_assignment_A):
            break

    while True:
        prev_assignment_B = assignment_B.copy()
        centers_B = reassign_centers(gaussians_B_vecs, current_num_centers, assignment_B, centers_B, xyz_weight, semantic_weight)
        assignment_B = group_points_by_knn_with_semantics(gaussians_B_vecs, centers_B, xyz_weight, semantic_weight)
        if np.all(assignment_B == prev_assignment_B):
            break

    # ----- Build Final Groups -----
    # For Group A, return the xyz centers (recover by dividing by xyz_weight).
    groups_A = [[] for _ in range(current_num_centers)]
    N_A = gaussians_A_vecs.shape[0]
    for i in range(N_A):
        groups_A[assignment_A[i]].append(i)
    centers_xyz_A = centers_A[:, :3] / xyz_weight
    center_features_A = centers_A[:, 3:] / semantic_weight
    # For Group B, return the full centers (both xyz and semantic parts).
    groups_B = [[] for _ in range(current_num_centers)]
    N_B = gaussians_B_vecs.shape[0]
    for i in range(N_B):
        groups_B[assignment_B[i]].append(i)
    centers_xyz_B = centers_B[:, :3]
    center_features_B = centers_B[:, 3:]
    return centers_xyz_A, center_features_A, groups_A, assignment_A, centers_xyz_B, center_features_B, groups_B, assignment_B



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
    group_xyz = []
    xyz = gaussians.get_xyz.detach().cpu().numpy()
    
    for group in groups:
        group_semantic = semantic_features[group]
        # avg_semantic = robust_semantic_aggregation(group_semantic)
        # avg_semantic = np.mean(group_semantic, axis=0)
        avg_semantic = geometric_median(group_semantic)
        group_features.append(avg_semantic)
        group_xyz.append(np.mean(xyz[group], axis=0))
    #     # print average of the entry in semantic_features
    #     print(f"Average of the entry in semantic_features: {np.mean(semantic_features)}")
    # # print average distance between points in each group
    # for i in range(len(group_features)):
    #     group = group_features[i].reshape(1, -1)
    #     if len(group) > 0:
    #         avg_dist = np.mean(np.linalg.norm(group_semantic - group, axis=1))
    #         print(f"Average distance between points in group in terms of features {i}: {avg_dist}")
    group_features = np.array(group_features)
    group_xyz = np.array(group_xyz)
    # # print shape of group_features
    # print(f"Shape of group_features: {group_features.shape}")
    return group_features, group_xyz

## LOCAL REFINEMENT START

def compute_unary_cost(semantic_1, semantic_2, alpha=1.0):
    """
    Compute the unary cost matrix based on semantic feature differences.
    
    Parameters:
        semantic_1: (n x k) numpy array for set 1.
        semantic_2: (n x k) numpy array for set 2.
        alpha: Weight on the semantic (unary) cost.
    
    Returns:
        cost_unary: (n x n) numpy array where cost_unary[i, j] = alpha * ||semantic_1[i] - semantic_2[j]||^2.
    """
    # Compute pairwise squared Euclidean distances between semantic features.
    cost_unary = np.sum((semantic_1[:, None, :] - semantic_2[None, :, :]) ** 2, axis=2)
    return alpha * cost_unary

def compute_neighbor_indices(xyz, k_nn=5):
    """
    Build a neighbor graph for a set of points using k-nearest neighbors.
    
    Parameters:
        xyz: (n x 3) numpy array of point coordinates.
        k_nn: Number of neighbors to use (excluding self).
    
    Returns:
        neighbors: (n x k_nn) array of indices, where neighbors[i] are the indices of the k_nn nearest neighbors of point i.
    """
    tree = cKDTree(xyz)
    # Query for k_nn+1 neighbors (the first neighbor is the point itself)
    _, indices = tree.query(xyz, k=k_nn + 1)
    # Remove the first column (self) for each point
    return indices[:, 1:]

def graph_hungarian_matching_with_smoothness_efficient(xyz_1, xyz_2, semantic_1, semantic_2,
                                              k_nn=5, alpha=1.0, lambda_s=1.0, iterations=10):
    """
    Perform graph matching with spatial smoothness constraints using vectorized operations.
    
    Parameters:
        xyz_1: (n x 3) numpy array of coordinates for set 1.
        xyz_2: (n x 3) numpy array of coordinates for set 2.
        semantic_1: (n x k) numpy array of semantic features for set 1.
        semantic_2: (n x k) numpy array of semantic features for set 2.
        k_nn: Number of nearest neighbors (spatial) to consider.
        alpha: Weight on the unary (semantic) cost.
        lambda_s: Weight on the smoothness (pairwise) cost.
        iterations: Number of iterative refinement steps.
        
    Returns:
        assignment: A 1D numpy array of length n such that assignment[i] = j means point i in set 1 is matched to point j in set 2.
    """
    n = xyz_1.shape[0]
    
    # Compute the unary cost (based solely on semantic features)
    cost_unary = compute_unary_cost(semantic_1, semantic_2, alpha=alpha)
    
    # Build neighbor indices for set 1 (using spatial information)
    neighbors_1 = compute_neighbor_indices(xyz_1, k_nn=k_nn)  # shape: (n, k_nn)
    
    # Get an initial matching using only the unary cost.
    row_ind, col_ind = linear_sum_assignment(cost_unary)
    assignment = np.zeros(n, dtype=int)
    assignment[row_ind] = col_ind
    
    # Precompute the distances in set 1 between each point and its neighbors.
    # For each i, compute distances to its k_nn neighbors.
    # xyz_1[neighbors_1] has shape (n, k_nn, 3) and xyz_1[:, None, :] has shape (n, 1, 3).
    d1 = np.linalg.norm(xyz_1[:, None, :] - xyz_1[neighbors_1], axis=2)  # shape: (n, k_nn)
    
    # Iteratively refine the matching by adding a spatial smoothness cost.
    for it in range(iterations):
        # Start with the unary cost.
        composite_cost = np.copy(cost_unary)
        
        # For each point i in set 1, add the smoothness cost for each candidate match j in set 2.
        # The idea: for each neighbor i' of i, compare the distance between i and i' in set 1 with the distance 
        # between candidate match j and the currently assigned match for i' in set 2.
        for i in range(n):
            for neighbor in neighbors_1[i]:
                # Distance between point i and its neighbor in set 1
                d1 = np.linalg.norm(xyz_1[i] - xyz_1[neighbor])
                # The currently assigned match in set 2 for the neighbor i'
                j_neighbor = assignment[neighbor]
                # For every candidate j (possible match for point i) in set 2, compute the distance between
                # candidate j and the match of the neighbor.
                # This produces a vector (of length n) for the candidate cost.
                d2 = np.linalg.norm(xyz_2 - xyz_2[j_neighbor], axis=1)
                # Compute squared difference between the spatial distances.
                smooth_cost = (d1 - d2) ** 2
                # Add the smoothness cost for this neighbor to all candidate matches for point i.
                composite_cost[i, :] += lambda_s * smooth_cost
        
        # Solve the updated assignment problem using the composite cost matrix.
        row_ind, col_ind = linear_sum_assignment(composite_cost)
        assignment = np.zeros(n, dtype=int)
        assignment[row_ind] = col_ind
        
        # (Optional) Print the total cost at each iteration.
        total_cost = composite_cost[row_ind, col_ind].sum()
        print(f"Iteration {it + 1}: Total cost = {total_cost:.4f}")
    
    return assignment
def graph_knn_matching_with_smoothness_efficient_symmetric(xyz_1, xyz_2, semantic_1, semantic_2,
                                                        k_nn=5, alpha=1.0, lambda_s=1.0, iterations=10):
    """
    Compute two matching assignments (set1 -> set2 and set2 -> set1) by incorporating 
    semantic similarity (unary cost) and spatial smoothness constraints using neighbor graphs.
    
    For assignment1to2: For each point in set1, the composite cost for candidate match j in set2 is:
        composite_cost_1[i, j] = cost_unary[i, j] 
             + lambda_s * sum_{i' in neighbors1[i]} ( ||xyz_1[i] - xyz_1[i']|| - ||xyz_2[j] - xyz_2[ assignment1to2[i'] ]|| )^2.
             
    For assignment2to1: For each point in set2, the composite cost for candidate match i in set1 is:
        composite_cost_2[i, j] = cost_unary[i, j] 
             + lambda_s * sum_{j' in neighbors2[j]} ( ||xyz_2[j] - xyz_2[j']|| - ||xyz_1[i] - xyz_1[ assignment2to1[j'] ]|| )^2.
             
    Parameters:
        xyz_1: (n x 3) numpy array for set 1.
        xyz_2: (n x 3) numpy array for set 2.
        semantic_1: (n x k) numpy array of semantic features for set 1.
        semantic_2: (n x k) numpy array of semantic features for set 2.
        k_nn: Number of neighbors to use for each set.
        alpha: Weight for the semantic (unary) cost.
        lambda_s: Weight for the smoothness (pairwise) cost.
        iterations: Number of iterative refinement steps.
    
    Returns:
        assignment1to2: 1D numpy array of length n such that assignment1to2[i] = j is the match for point i in set 1.
        assignment2to1: 1D numpy array of length n such that assignment2to1[j] = i is the match for point j in set 2.
    """
    n = xyz_1.shape[0]
    
    # Step 1: Compute the unary cost matrix.
    cost_unary = compute_unary_cost(semantic_1, semantic_2, alpha=alpha)
    
    # Step 2: Build neighbor graphs for each set.
    neighbors1 = compute_neighbor_indices(xyz_1, k_nn=k_nn)
    neighbors2 = compute_neighbor_indices(xyz_2, k_nn=k_nn)
    
    # Step 3: Initialize assignments using the nearest neighbor in terms of the unary cost.
    # For set1 -> set2: For each point i in set1, choose j minimizing cost_unary[i, :].
    assignment1to2 = np.argmin(cost_unary, axis=1)
    # For set2 -> set1: For each point j in set2, choose i minimizing cost_unary[:, j].
    assignment2to1 = np.argmin(cost_unary, axis=0)
    
    # Iteratively refine both assignments.
    for it in range(iterations):
        # --- Update assignment from set1 to set2 using neighbors1 ---
        composite_cost_1 = np.copy(cost_unary)
        for i in range(n):
            # For each neighbor of point i in set1...
            for neighbor in neighbors1[i]:
                # Distance between point i and its neighbor in set1.
                d1 = np.linalg.norm(xyz_1[i] - xyz_1[neighbor])
                # Get the currently assigned match for the neighbor.
                j_neighbor = assignment1to2[neighbor]
                # For every candidate j in set2, compute the distance between candidate j and the neighbor's match.
                d2 = np.linalg.norm(xyz_2 - xyz_2[j_neighbor], axis=1)  # shape: (n,)
                # Compute the smoothness penalty.
                smooth_cost = (d1 - d2) ** 2
                # Add the weighted smoothness cost to candidate matches for point i.
                composite_cost_1[i, :] += lambda_s * smooth_cost
        
        # For each point in set1, update the assignment by choosing the candidate in set2 with minimal composite cost.
        assignment1to2 = np.argmin(composite_cost_1, axis=1)
        
        # --- Update assignment from set2 to set1 using neighbors2 ---
        composite_cost_2 = np.copy(cost_unary)
        for j in range(n):
            # For each neighbor of point j in set2...
            for neighbor in neighbors2[j]:
                # Distance between point j and its neighbor in set2.
                d1 = np.linalg.norm(xyz_2[j] - xyz_2[neighbor])
                # Get the currently assigned match for the neighbor in set2.
                i_neighbor = assignment2to1[neighbor]
                # For every candidate i in set1, compute the distance between candidate i and the neighbor's match.
                d2 = np.linalg.norm(xyz_1 - xyz_1[i_neighbor], axis=1)  # shape: (n,)
                smooth_cost = (d1 - d2) ** 2
                # Add the weighted smoothness cost to candidate matches for point j.
                composite_cost_2[:, j] += lambda_s * smooth_cost
        
        # For each point in set2, update the assignment by choosing the candidate in set1 with minimal composite cost.
        assignment2to1 = np.argmin(composite_cost_2, axis=0)
        
        # (Optional) Print current iteration costs.
        total_cost1 = np.sum(composite_cost_1[np.arange(n), assignment1to2])
        total_cost2 = np.sum(composite_cost_2[assignment2to1, np.arange(n)])
        print(f"Iteration {it + 1}: Total cost1 = {total_cost1:.4f}, Total cost2 = {total_cost2:.4f}")
    
    return assignment1to2, assignment2to1


from sklearn.metrics import pairwise_distances

def build_adjacency_matrix(centroids, radius=0.5):
    """
    Build an adjacency matrix for the given centroids.
    Two clusters i, j are considered adjacent if their distance < radius.
    """
    dist_mat = pairwise_distances(centroids, centroids, metric='euclidean')
    adjacency = (dist_mat < radius).astype(np.float32)
    
    # Remove self-adjacency if desired:
    np.fill_diagonal(adjacency, 0.0)
    return adjacency

def build_descriptors(centroids, features, lambda_geom=1.0):
    """
    Concatenate scaled centroids and features into a single descriptor.
    centroids: (N,3) array
    features: (N,d) array
    Return: descriptors (N, 3 + d)
    """
    # Scale the geometry part by lambda_geom
    scaled_centroids = lambda_geom * centroids
    descriptors = np.concatenate([scaled_centroids, features], axis=1)
    return descriptors

def initial_cost_matrix(desc1, desc2):
    """
    Compute a simple cost matrix based on the squared Euclidean distance
    of descriptors between set1 and set2.
    desc1: (N1, D), desc2: (N2, D)
    Return: (N1, N2) cost matrix
    """
    dists = pairwise_distances(desc1, desc2, metric='euclidean')
    return dists**2

def adjacency_penalty_matrix(adjacency1, adjacency2, current_matches):
    """
    Compute a penalty matrix based on adjacency mismatches.

    adjacency1: (N1, N1) adjacency for set1
    adjacency2: (N2, N2) adjacency for set2
    current_matches: array of shape (N1,) where current_matches[i] = j 
                     means cluster i in set1 is matched to cluster j in set2.
                     If a cluster i in set1 is unmatched, might store -1 or np.nan.

    Return:
    penalty_matrix: (N1, N2), penalty_matrix[i, j] = penalty if i matched with j
    """
    N1 = adjacency1.shape[0]
    N2 = adjacency2.shape[0]
    
    # Initialize all penalties to 0
    penalty_mat = np.zeros((N1, N2), dtype=np.float32)
    
    # For each cluster i in set1, we know it's matched to j = current_matches[i].
    # We'll penalize any mismatch with neighbors. 
    for i in range(N1):
        j_matched = current_matches[i]
        if j_matched < 0:
            continue
        
        # For all neighbors p of i in set1, see if p is matched to q in set2
        # and check adjacency mismatch in set2.
        neighbors_i = np.where(adjacency1[i] > 0)[0]
        for p in neighbors_i:
            q_matched = current_matches[p]
            if q_matched < 0:
                continue
            
            # If i and p are neighbors in set1 but j_matched and q_matched 
            # are NOT neighbors in set2, add penalty.
            if adjacency2[j_matched, q_matched] == 0:
                # You can define your penalty function here. Let's do a constant + 
                # optional distance-based penalty. For simplicity, use constant.
                penalty_mat[i, j_matched] += 1.0  # increment a mismatch penalty
    
    return penalty_mat

def iterative_refinement(desc1, desc2, adjacency1, adjacency2, 
                         alpha=1.0, beta=1.0, max_iter=10):
    """
    Perform iterative refinement of a bipartite matching with adjacency constraints.
    
    desc1: (N1, D) descriptors for set1
    desc2: (N2, D) descriptors for set2
    adjacency1: (N1, N1)
    adjacency2: (N2, N2)
    alpha, beta: weights for descriptor vs. adjacency penalty
    max_iter: number of refinement iterations
    
    Returns:
    final_matches: array of shape (N1,) where final_matches[i] = j
    """
    N1 = desc1.shape[0]
    N2 = desc2.shape[0]
    
    # If the sets have different sizes, you might handle that with dummy clusters 
    # or partial matching. For simplicity, assume N1 == N2 here:
    assert N1 == N2, "For a simple 1-to-1 matching, we require same number of clusters."
    
    # Compute initial cost matrix (descriptor-based)
    cost_mat = initial_cost_matrix(desc1, desc2)
    
    # Iterative update
    current_matches = None
    for iteration in range(max_iter):
        # Solve the assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_mat)
        
        # row_ind[i] = the i-th matched row index
        # col_ind[i] = the j-th matched col index
        # We want final_matches[row_ind[i]] = col_ind[i]
        new_matches = -1 * np.ones(N1, dtype=int)
        for i_r, j_c in zip(row_ind, col_ind):
            new_matches[i_r] = j_c
        # print how many assignments changed
        if current_matches is not None:
            diff = np.sum(new_matches != current_matches)
            print(f"Iteration {iteration}: {diff} assignments changed.")
        # If matches haven't changed, we can stop
        if current_matches is not None and np.all(new_matches == current_matches):
            print(f"Converged at iteration {iteration}")
            break
        
        current_matches = new_matches
        
        # Compute adjacency penalty based on current_matches
        pen_mat = adjacency_penalty_matrix(adjacency1, adjacency2, current_matches)
        
        # Update the cost matrix with new iteration:
        desc_cost = initial_cost_matrix(desc1, desc2)
        cost_mat = alpha * desc_cost + beta * pen_mat

        # print total cost
        print(f"Iteration {iteration}: Total cost = {cost_mat[row_ind, col_ind].sum():.4f}")
    
    return current_matches


## LOCAL REFINEMENT END


def match_groups_and_points(gaussians1, gaussians2, num_centers, xyz_weight, semantic_weight, method, group_pkl_paths, color_by_group, use_joint_fps_and_knn_grouping):
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
    if use_joint_fps_and_knn_grouping:
            group_pkl_path_1 = "joint_" + group_pkl_path_1
            group_pkl_path_2 = "joint_" + group_pkl_path_2
    # check if the two paths exist
    if not os.path.exists(group_pkl_path_1) or not os.path.exists(group_pkl_path_2):
        if use_joint_fps_and_knn_grouping:
            print("Using joint fps and knn grouping")
            # centers_xyz1, center_features1, groups1, assignment1, centers_xyz2, center_features2, groups2, assignment2 = joint_fps_and_knn_grouping(
            #     gaussians1, gaussians2, num_centers, xyz_weight=xyz_weight, semantic_weight=semantic_weight
            # )
            centers_xyz1, center_features1, groups1, assignment1 = your_fps_and_knn_grouping(gaussians1, num_centers, xyz_weight=xyz_weight, semantic_weight=semantic_weight)
            centers_xyz2, center_features2, groups2, assignment2 = your_fps_and_knn_grouping_with_reference_centers(gaussians2, num_centers, centers_features_reference=center_features1, xyz_weight=xyz_weight, semantic_weight=semantic_weight)
        else:
            print("Using separate fps and knn grouping")
            centers_xyz1, center_features1, groups1, assignment1 = your_fps_and_knn_grouping(gaussians1, num_centers, xyz_weight=xyz_weight, semantic_weight=semantic_weight)
            centers_xyz2, center_features2, groups2, assignment2 = your_fps_and_knn_grouping(gaussians2, num_centers, xyz_weight=xyz_weight, semantic_weight=semantic_weight)
        # save centers_xyz1 and centers_xyz2 to ply files
        write_points_to_ply(centers_xyz1, "furthest_sampled_gaussians1.ply")
        write_points_to_ply(centers_xyz2, "furthest_sampled_gaussians2.ply")

        # save groups1 and groups2 which are list of lists
        with open(group_pkl_path_1, "wb") as f:
            pickle.dump(groups1, f)
        with open(group_pkl_path_2, "wb") as f:
            pickle.dump(groups2, f)
    else:
        print("--------------------------------")
        print(f"Clustering already done. Loading groups from {group_pkl_path_1} and {group_pkl_path_2}")
        print("--------------------------------")
        # load groups1 and groups2
        with open(group_pkl_path_1, "rb") as f:
            groups1 = pickle.load(f)
        with open(group_pkl_path_2, "rb") as f:
            groups2 = pickle.load(f)
    
    
    # Compute average semantic features for each group
    group_features1, group_xyz1 = compute_group_features(gaussians1, groups1)
    group_features2, group_xyz2 = compute_group_features(gaussians2, groups2)

    # Color the gaussians by group
    if color_by_group:
        print("Coloring gaussians by group")
        color_gaussians_by_group(gaussians1, groups1)
        color_gaussians_by_group(gaussians2, groups2)


    # # print shape of group_features1 and group_features2
    # print(f"Shape of group_features1: {group_features1.shape}")
    # print(f"Shape of group_features2: {group_features2.shape}")
    
    # Option 1: Use Hungarian algorithm to find optimal one-to-one matching
    if method == "hungarian" or method == "local_consistency_hungarian":
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
        elif method == "local_consistency_hungarian":
            alpha = 1.0/np.sqrt(group_features1.shape[1])
            lambda_s = 10.0/np.sqrt(group_xyz1.shape[1])
            gamma = 100
            # First
            # assignment = graph_matching_with_neighbor_agreement(
            #     group_xyz1, group_xyz2, group_features1, group_features2,
            #     num_neighbors=5, target_percentage=0.6, alpha=alpha, beta=lambda_s, gamma=gamma, max_iter=100, tol=1e-5
            # )
            
            # Second
            assignment = graph_hungarian_matching_with_smoothness_efficient(
                group_xyz1, group_xyz2, group_features1, group_features2,
                k_nn=5, alpha=alpha, lambda_s=lambda_s, iterations=100
            )

            # Third
            # lambda_geom = (5.0/np.sqrt(group_xyz1.shape[1]))/(1.0/np.sqrt(group_features1.shape[1]))
            # # build descriptors using features and geometry
            # desc1 = build_descriptors(group_xyz1, group_features1, lambda_geom=lambda_geom)
            # desc2 = build_descriptors(group_xyz2, group_features2, lambda_geom=lambda_geom)
            # # Build adjacency (using small radius to force few neighbors)
            # radius = 0.5
            # adjacency1 = build_adjacency_matrix(group_xyz1, radius=radius)
            # adjacency2 = build_adjacency_matrix(group_xyz2, radius=radius)
            # assignment = iterative_refinement(
            #     desc1, desc2, adjacency1, adjacency2,
            #     alpha=alpha, beta=lambda_s, max_iter=100
            # )
            row_ind = np.arange(len(groups1))
            col_ind = assignment
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
            feat1_semantic = gaussians1.get_semantic_feature[points1].detach().cpu().numpy()
            feat2_semantic = gaussians2.get_semantic_feature[points2].detach().cpu().numpy()
            feat1_semantic = feat1_semantic.squeeze(1)
            feat2_semantic = feat2_semantic.squeeze(1)
            feat1_xyz = gaussians1.get_xyz[points1].detach().cpu().numpy()
            feat2_xyz = gaussians2.get_xyz[points2].detach().cpu().numpy()
            feat1_scale = gaussians1.get_scaling[points1].detach().cpu().numpy()
            feat2_scale = gaussians2.get_scaling[points2].detach().cpu().numpy()
            feat1_opacity = gaussians1.get_opacity[points1].detach().cpu().numpy()
            feat2_opacity = gaussians2.get_opacity[points2].detach().cpu().numpy()
            semantic_weight = 1.0/np.sqrt(feat1_semantic.shape[1])
            xyz_weight = 1.0/np.sqrt(feat1_xyz.shape[1])
            scale_weight = 2.0/np.sqrt(feat1_scale.shape[1])
            opacity_weight = 2.0/np.sqrt(feat1_opacity.shape[1])
            feat1 = np.concatenate([semantic_weight * feat1_semantic, xyz_weight * feat1_xyz, scale_weight * feat1_scale, opacity_weight * feat1_opacity], axis=1)
            feat2 = np.concatenate([semantic_weight * feat2_semantic, xyz_weight * feat2_xyz, scale_weight * feat2_scale, opacity_weight * feat2_opacity], axis=1)
            
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
    elif method == "knn" or method == "local_consistency_knn":
        res = faiss.StandardGpuResources()  # create GPU resources
        if method == "knn":
            # Create FAISS indices for both sets of group features
            d = group_features1.shape[1]  # dimensionality of features
            
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
        elif method == "local_consistency_knn":
            alpha = 1.0/np.sqrt(group_features1.shape[1])
            lambda_s = 10.0/np.sqrt(group_xyz1.shape[1])
            assignment_1to2, assignment_2to1 = graph_knn_matching_with_smoothness_efficient_symmetric(
                group_xyz1, group_xyz2, group_features1, group_features2,
                k_nn=10, alpha=alpha, lambda_s=lambda_s, iterations=100
            )
            group_matching_1to2 = assignment_1to2
            group_matching_2to1 = assignment_2to1

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
    
    xyz_weight = 5.0/np.sqrt(3)
    semantic_weight = 1.0/np.sqrt(256)

    # print out the time of execution
    start_time = time.time()
    group_pkl_paths = args.group_pkl_paths
    group_matching_method = args.group_matching_method
    color_by_group = args.color_by_group
    num_centers = args.num_centers
    new_gaussians1, new_gaussians2, new_matches = match_groups_and_points(matcher.gaussians1, matcher.gaussians2, num_centers=num_centers, xyz_weight=xyz_weight, semantic_weight=semantic_weight, method=group_matching_method, group_pkl_paths=group_pkl_paths, color_by_group=color_by_group, use_joint_fps_and_knn_grouping=args.use_joint_fps_and_knn_grouping)
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
