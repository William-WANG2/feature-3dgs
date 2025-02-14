import gc
import os
import pickle
import sys
from scene import Scene, GaussianModel
from argparse import ArgumentParser
from arguments import ModelParamsMatching, PipelineParams
import torch
import faiss
import numpy as np
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


# dirty fix for the import error of the original instant_nsr_pl package
sys.path.append(os.path.abspath("./instant_nsr_pl"))
from instant_nsr_pl.models.neus import NeuSModel
import systems
from instant_nsr_pl.utils.misc import load_config

class KNNMatchingInterpolation:
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
        
        # Store pipeline parameters
        self.pipe = pipe

        # Store placeholders for neus model ckpt and config path
        self.neus_model_ckpt_1 = None
        self.neus_config_path_1 = None
        self.neus_model_ckpt_2 = None
        self.neus_config_path_2 = None

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

        # Add matching_features
        parser.add_argument('--matching_features', nargs='+', type=str, default=['semantic_feature', 'dir_centers'], choices=['semantic_feature', 'xyz', 'sr_features', 'dir_centers', 'sdf'])

        # Add interpolation_features
        parser.add_argument('--interpolation_features', nargs='+', type=str, default=['xyz', 'features_dc', 'features_rest', 'scaling', 'opacity', 'rotation'], choices=['xyz', 'features_dc', 'features_rest', 'scaling', 'opacity', 'rotation', 'semantic_feature'])

        # Add neus model ckpt and config path
        parser.add_argument('--neus_model_ckpt_1', type=str, default='./exp/neus-blender-dog/example@20250112-191501/ckpt/epoch=0-step=20000.ckpt')
        parser.add_argument('--neus_config_path_1', type=str, default='./instant_nsr_pl/configs/neus-blender.yaml')
        parser.add_argument('--neus_model_ckpt_2', type=str, default='./exp/neus-blender-cat/example@20250110-230313/ckpt/epoch=0-step=20000.ckpt')
        parser.add_argument('--neus_config_path_2', type=str, default='./instant_nsr_pl/configs/neus-blender.yaml')
        
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
    
    def get_gaussian_sdf(self, xyz, neus_model_ckpt, config_path):
        """
            Get SDF of each GaussianModel from NeusModel
            Args:
                xyz: xyz coordinates of the points to get SDF
                neus_model_ckpt: path to the NeusModel checkpoint
                config_path: path to the NeusModel config file
        """
        config = load_config(config_path)
        neus_model = systems.make('neus-system', config, neus_model_ckpt)
        geometry = neus_model.model.geometry
        sdf = geometry.forward(xyz, with_grad=False, with_feature=False, with_laplace=False)
        return sdf

    def compute_bidirectional_knn(self, k=1, features_to_use=['semantic_feature'], num_nearest_neighbors=20):
        """
        Compute bidirectional KNN matching between two GaussianModels using multiple features
        Args:
            k: Number of nearest neighbors for high-dimensional feature matching (default=1)
            features_to_use: List of feature names to use for matching (e.g., ['semantic_feature', 'xyz'])
            num_nearest_neighbors: Number of nearest neighbors for position matching (default=20)

        Returns:
            matches_1to2: Indices of nearest neighbors in gaussians2 for each point in gaussians1 dim: (N1, k)
            matches_2to1: Indices of nearest neighbors in gaussians1 for each point in gaussians2 dim: (N2, k)
            nearest_neighbors_1to1: Indices of nearest neighbors in gaussians1 for each point in gaussians1 dim: (N1, num_nearest_neighbors)
            nearest_neighbors_2to2: Indices of nearest neighbors in gaussians2 for each point in gaussians2 dim: (N2, num_nearest_neighbors)
            features1: Features of gaussians1 dim: (N1, D)
            features2: Features of gaussians2 dim: (N2, D)
            xyz_1: xyz coordinates of gaussians1 dim: (N1, 3)
            xyz_2: xyz coordinates of gaussians2 dim: (N2, 3)
        """
        features1_list = []
        features2_list = []
        
        for feature in features_to_use:
            if feature == 'semantic_feature':
                feat1 = self.gaussians1.get_semantic_feature.detach().squeeze(-1).cpu().numpy()
                feat2 = self.gaussians2.get_semantic_feature.detach().squeeze(-1).cpu().numpy()
                print("min, max of semantic_feature_1: ", feat1.min(), feat1.max())
                print("min, max of semantic_feature_2: ", feat2.min(), feat2.max())
                feat1 = feat1.squeeze(1)
                feat2 = feat2.squeeze(1)
                weight = 2 / np.sqrt(feat1.shape[1])
                feat1 = feat1 * weight
                feat2 = feat2 * weight
            elif feature == 'xyz':
                feat1 = self.gaussians1.get_xyz.detach().cpu().numpy()
                feat2 = self.gaussians2.get_xyz.detach().cpu().numpy()
                weight = 1 / np.sqrt(feat1.shape[1])
                feat1 = feat1 * weight
                feat2 = feat2 * weight
            elif feature == 'sdf':
                xyz_1 = self.gaussians1.get_xyz
                xyz_2 = self.gaussians2.get_xyz
                sdf_1 = self.get_gaussian_sdf(xyz_1, self.neus_model_ckpt_1, self.neus_config_path_1).unsqueeze(-1)
                sdf_2 = self.get_gaussian_sdf(xyz_2, self.neus_model_ckpt_2, self.neus_config_path_2).unsqueeze(-1)
                print("abs min, max of sdf_1: ", sdf_1.abs().min(), sdf_1.abs().max())
                print("abs min, max of sdf_2: ", sdf_2.abs().min(), sdf_2.abs().max())
                feat1 = sdf_1.detach().cpu().numpy()
                feat2 = sdf_2.detach().cpu().numpy()
                weight = 5 / np.sqrt(feat1.shape[1])
                feat1 = feat1 * weight
                feat2 = feat2 * weight
            elif feature == 'sr_features':
                feat1 = self.gaussians1.get_features.flatten(start_dim=1).detach().cpu().numpy()
                feat2 = self.gaussians2.get_features.flatten(start_dim=1).detach().cpu().numpy()
                weight = 1 / np.sqrt(feat1.shape[1])
                feat1 = feat1 * weight
                feat2 = feat2 * weight
            elif feature == 'dir_centers':
                xyz_center_1 = self.gaussians1.get_xyz.mean(dim=0).detach().cpu().numpy()
                xyz_center_2 = self.gaussians2.get_xyz.mean(dim=0).detach().cpu().numpy()
                xyzs_1 = self.gaussians1.get_xyz.detach().cpu().numpy()
                xyzs_2 = self.gaussians2.get_xyz.detach().cpu().numpy()
                feat1 = xyzs_1 - xyz_center_1
                feat2 = xyzs_2 - xyz_center_2
                print("min, max of dir_centers_1: ", feat1.min(), feat1.max())
                print("min, max of dir_centers_2: ", feat2.min(), feat2.max())
                weight = 1 / np.sqrt(feat1.shape[1])
                feat1 = feat1 * weight
                feat2 = feat2 * weight
            else:
                raise ValueError(f"Invalid feature to use: {feature}")
            
            features1_list.append(feat1)
            features2_list.append(feat2)
        
        # Concatenate all features
        features1 = np.concatenate(features1_list, axis=1)
        features2 = np.concatenate(features2_list, axis=1)
        
        # Ensure features are float32 (required by faiss)
        features1 = features1.astype(np.float32)
        features2 = features2.astype(np.float32)
        
        print(f"Using features for KNN matching: {features_to_use}")
        print("Combined features1 shape: ", features1.shape)
        print("Combined features2 shape: ", features2.shape)
        
        # Create faiss index for fast KNN
        d = features1.shape[1]  # feature dimension
        print("feature dimension: ", d)
        
        # Use GPU if available
        if torch.cuda.is_available():
            print("Using GPU for KNN matching")
            res = faiss.StandardGpuResources()
            index1 = faiss.GpuIndexFlatL2(res, d)
            index2 = faiss.GpuIndexFlatL2(res, d)
        else:
            print("Using CPU for KNN matching")
            index1 = faiss.IndexFlatL2(d)
            index2 = faiss.IndexFlatL2(d)

        # Build indices
        index1.add(features1)
        index2.add(features2)

        # Compute KNN in both directions
        distances_2to1, matches_2to1 = index1.search(features2, k)  # N2 -> N1
        distances_1to2, matches_1to2 = index2.search(features1, k)  # N1 -> N2
        # print the max, min, median, mean  of distances_2to1 and distances_1to2
        print(f"Max distance 2to1: {distances_2to1.max()}")
        print(f"Min distance 2to1: {distances_2to1.min()}")
        print(f"Max distance 1to2: {distances_1to2.max()}")
        print(f"Min distance 1to2: {distances_1to2.min()}")
        print(f"Median distance 2to1: {np.median(distances_2to1)}")
        print(f"Median distance 1to2: {np.median(distances_1to2)}")
        print(f"Mean distance 2to1: {np.mean(distances_2to1)}")
        print(f"Mean distance 1to2: {np.mean(distances_1to2)}")

        # find nearest neighbors for each point in gaussians1 and gaussians2 using positions
        xyz_1 = self.gaussians1.get_xyz.detach().cpu().numpy()
        xyz_2 = self.gaussians2.get_xyz.detach().cpu().numpy()
        
        d = xyz_1.shape[1]
        xyz_1 = xyz_1.astype(np.float32)
        xyz_2 = xyz_2.astype(np.float32)
        index1 = faiss.IndexFlatL2(d)
        index2 = faiss.IndexFlatL2(d)
        index1.add(xyz_1)
        index2.add(xyz_2)
        distances_1to1, nearest_neighbors_1to1 = index1.search(xyz_1, num_nearest_neighbors)  # N1 -> N1
        distances_2to2, nearest_neighbors_2to2 = index2.search(xyz_2, num_nearest_neighbors)  # N2 -> N2

        return matches_1to2, matches_2to1, nearest_neighbors_1to1, nearest_neighbors_2to2, features1, features2, xyz_1, xyz_2

    def soft_matching_optimization_minibatch(
        self,
        feat1,          # (N1, D)
        feat2,          # (N2, D)
        xyz2,           # (N2, 3)
        matches_1to2,   # (N1, 1)
        neighbor_ids,   # list of neighbors for each i in [0..N1-1], shape: (N1, k)
        batch_size=1,
        num_epochs=100,
        lr=10,
        lambda_smooth=1.0,
        lambda_entropy=0.1
    ):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)

        # Move data to CPU first (or keep it on CPU if too large)
        feat1_cpu = torch.tensor(feat1, dtype=torch.float32, device='cpu')
        feat2_cpu = torch.tensor(feat2, dtype=torch.float32, device='cpu')
        xyz2_cpu  = torch.tensor(xyz2, dtype=torch.float32, device='cpu')

        N1, D = feat1_cpu.shape
        N2, _ = feat2_cpu.shape

        print("N1, D: ", N1, D)
        print("N2, _: ", N2, _)

        # -------------------------------------------
        # Create M on CPU as a normal (non-Parameter)
        # -------------------------------------------
        M_cpu = 0.2 * torch.ones((N1, N2), dtype=torch.float32, device='cpu')
        for i in range(N1):
            nn_idx = matches_1to2[i, 0]  # the known nearest neighbor
            M_cpu[i, nn_idx] = 1.0

        # For demonstration, do the full cdist on CPU (WARNING: big if N2 is large!)
        dist_xyz2_cpu = torch.cdist(xyz2_cpu, xyz2_cpu, p=2)  # (N2, N2)
        dist_xyz2_gpu = dist_xyz2_cpu.to(device)

        # We'll create a simple DataLoader over row indices [0..N1-1].
        dataset = torch.arange(N1)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        for epoch in range(num_epochs):
            # We do a manual "accumulate then update" approach. 
            # In this example, we will do an update on each batch directly, 
            # but you could also accumulate across an entire epoch if you wish.

            total_loss = 0.0

            num_batches_pass = 0

            for batch_rows_cpu in loader:
                num_batches_pass += 1
                if num_batches_pass % 100 == 0:
                    print(f"Passing batch {num_batches_pass}")

                # Ensure these row indices are on CPU
                batch_rows_cpu = batch_rows_cpu.to('cpu')  # shape (batch_size,)
                # Convert them to a list or CPU tensor
                batch_rows_list = batch_rows_cpu.tolist()

                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # 1) Copy relevant rows of M to GPU for forward/backward
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                # gather the needed feat1 rows on GPU:
                feat1_batch_gpu = feat1_cpu[batch_rows_list, :].to(device)

                # For the smoothness part, we also need the union_rows
                neighbor_set = set()
                for i_ in batch_rows_list:
                    for nk in neighbor_ids[i_]:
                        neighbor_set.add(nk)
                union_rows_list = list(set(batch_rows_list).union(neighbor_set))
                union_rows_list = sorted(union_rows_list)  # for consistency
                batch_rows_list_local_idx_in_union_row_list = [union_rows_list.index(i) for i in batch_rows_list]
                # print("batch_rows_list: ", batch_rows_list)
                # print("batch_rows_list_local_idx_in_union_row_list: ", batch_rows_list_local_idx_in_union_row_list)
                # print("union_rows_list: ", union_rows_list)

                # Now gather subM_union (rows in union_rows_list)
                subM_union_gpu = M_cpu[union_rows_list, :].clone().detach().to(device)
                subM_union_gpu.requires_grad_(True)

                subM_batch_gpu = subM_union_gpu[batch_rows_list_local_idx_in_union_row_list, :]

                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # 2) Forward pass (feature loss, smoothness, entropy)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                # a) Feature loss for subM_batch_gpu
                subM_batch_normed = torch.nn.functional.softmax(subM_batch_gpu, dim=1)
                # diff_feat shape: (batch_size, N2, D)
                diff_feat = feat1_batch_gpu.unsqueeze(1) - feat2_cpu.to(device).unsqueeze(0)
                cost_feat = torch.sum(diff_feat ** 2, dim=-1)  # (batch_size, N2)
                L_feat_batch = torch.sum(subM_batch_normed * cost_feat)

                # b) Smoothness for subM_batch + neighbors
                #    We only need the adjacency among union_rows. 
                #    But for the final C_sub, it’s subM_batch_normed^T A_sub subM_union_normed
                #    which yields an (N2 x N2) matrix we multiply by dist_xyz2. 
                #    That can still be huge if N2 is large, so be careful in practice!

                subM_union_normed = torch.nn.functional.softmax(subM_union_gpu, dim=1)

                # Build adjacency for the local indexing of union_rows
                row_map = {r: idx for idx, r in enumerate(union_rows_list)}
                # A_sub: shape (batch_size, len(union_rows_list))
                A_sub = torch.zeros((len(batch_rows_list), len(union_rows_list)), 
                                    device=device, dtype=torch.float32)
                for local_i, global_i in enumerate(batch_rows_list):
                    for nk in neighbor_ids[global_i]:
                        if nk in row_map:
                            A_sub[local_i, row_map[nk]] = 1.0

                # We also need subM_batch_normed as shape (N2, batch_size) for the product
                # so let's be consistent:
                subM_batch_normed_t = subM_batch_normed.transpose(0,1)  # (N2, batch_size)

                # subM_union_normed: (|union_rows|, N2)
                # We want C_sub = subM_batch_normed^T * A_sub * subM_union_normed 
                # => shape (N2, N2)
                #   * then multiply by dist_xyz2
                #   * sum up

                # But note: (N2, batch_size) x (batch_size, |union_rows|) => (N2, |union_rows|)
                # Then => (N2, |union_rows|) x (|union_rows|, N2) => (N2, N2)
                C_sub = subM_batch_normed_t @ A_sub @ subM_union_normed

                L_smooth_sub = lambda_smooth * torch.sum(C_sub * dist_xyz2_gpu)

                # c) Entropy loss for subM_batch
                eps = 1e-8
                # subM_batch_normed * log(subM_batch_normed)
                L_entropy_batch = torch.sum(subM_batch_normed * torch.log(subM_batch_normed + eps))
                L_entropy_batch = -lambda_entropy * L_entropy_batch

                # Combine
                L_total = L_feat_batch + L_smooth_sub + L_entropy_batch

                # ~~~~~~~~~~~~~~~~~~~~~
                # 3) Backward pass
                # ~~~~~~~~~~~~~~~~~~~~~
                L_total.backward()

                # ~~~~~~~~~~~~~~~~~~~~~
                # 4) Manual update
                # ~~~~~~~~~~~~~~~~~~~~~

                # We also need to update the union slice for the smoothness part if we want
                # consistent gradients on subM_union_gpu as well.
                # subM_union contains subM_batch.
                # So let's do that too:
                with torch.no_grad():
                    grad_union = subM_union_gpu.grad

                    # perform an SGD step on subM_union_gpu
                    # subM_union_gpu = subM_union_gpu - lr * grad_union
                    if grad_union is not None:
                        grad_union_cpu = grad_union.to('cpu')
                        # The rows of subM_union_gpu correspond to union_rows_list
                        M_cpu[union_rows_list, :] -= lr * grad_union_cpu

                total_loss += L_total.item()

            print("grad_union_cpu shape, max: ", grad_union_cpu.shape, grad_union_cpu.max())
            print(f"[Epoch {epoch+1}/{num_epochs}] Avg Loss = {total_loss / len(loader):.4f}")

        # After training, do a final softmax on CPU (or GPU in chunks)
        # M_cpu is our final matching matrix. Let's do a row-wise softmax and argmax:
        with torch.no_grad():
            # For large N1*N2, do it in chunks if needed
            M_final = []
            chunk_size = 1024
            for start in range(0, N1, chunk_size):
                end = min(start+chunk_size, N1)
                M_chunk = M_cpu[start:end, :]  # shape (chunk_size, N2)
                M_chunk_normed = torch.softmax(M_chunk, dim=1)
                chunk_matches = torch.argmax(M_chunk_normed, dim=1)
                M_final.append(chunk_matches)
            matches_1to2_out = torch.cat(M_final, dim=0)

        return matches_1to2_out.cpu().numpy()

    def create_matched_gaussians(self, matches_1to2, matches_2to1):
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
        k_values_1 = np.zeros(self.gaussians1.get_xyz.shape[0], dtype=np.int64)
        k_values_2 = np.zeros(self.gaussians2.get_xyz.shape[0], dtype=np.int64)
        
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
        current_idx_1 = np.zeros(self.gaussians1.get_xyz.shape[0], dtype=np.int64)
        current_idx_2 = np.zeros(self.gaussians2.get_xyz.shape[0], dtype=np.int64)
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
            'xyz': self.gaussians1.get_xyz[indices_1],
            'features_dc': self.gaussians1._features_dc[indices_1],
            'features_rest': self.gaussians1._features_rest[indices_1],
            'scaling': self.gaussians1._scaling[indices_1],
            'rotation': self.gaussians1._rotation[indices_1],
            'opacity': self.gaussians1.inverse_opacity_activation(
                self.gaussians1.opacity_activation(self.gaussians1._opacity[indices_1]) / k_values_1[indices_1].reshape(-1, 1)
            ),
            'semantic_feature': self.gaussians1._semantic_feature[indices_1],
            'active_sh_degree': self.gaussians1.active_sh_degree
        }
        
        new_gaussians2 = {
            'xyz': self.gaussians2.get_xyz[indices_2],
            'features_dc': self.gaussians2._features_dc[indices_2],
            'features_rest': self.gaussians2._features_rest[indices_2],
            'scaling': self.gaussians2._scaling[indices_2],
            'rotation': self.gaussians2._rotation[indices_2],
            'opacity': self.gaussians2.inverse_opacity_activation(
                self.gaussians2.opacity_activation(self.gaussians2._opacity[indices_2]) / k_values_2[indices_2].reshape(-1, 1)
            ),
            'semantic_feature': self.gaussians2._semantic_feature[indices_2],
            'active_sh_degree': self.gaussians2.active_sh_degree
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

        self.new_gaussians1 = new_gaussians1
        self.new_gaussians2 = new_gaussians2
        self.new_matches = new_matches

        print("Number of points in new_gaussians1: ", new_gaussians1['xyz'].shape[0])
        print("Number of points in new_gaussians2: ", new_gaussians2['xyz'].shape[0])

        return new_gaussians1, new_gaussians2, new_matches

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self.new_gaussians1['features_dc'].shape[1]*self.new_gaussians1['features_dc'].shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self.new_gaussians1['features_rest'].shape[1]*self.new_gaussians1['features_rest'].shape[2]):
            l.append('f_rest_{}'.format(i))

        l.append('opacity')
        for i in range(self.new_gaussians1['scaling'].shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self.new_gaussians1['rotation'].shape[1]):
            l.append('rot_{}'.format(i))
        # Add semantic features
        for i in range(self.new_gaussians1['semantic_feature'].shape[1]*self.new_gaussians1['semantic_feature'].shape[2]):  
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

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
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
    

    def export_new_gaussians(self, path_gaussian1, path_gaussian2, is_interpolation=False):
        self.save_new_gaussians(self.new_gaussians1, path_gaussian1, is_interpolation)
        self.save_new_gaussians(self.new_gaussians2, path_gaussian2, is_interpolation)


    def linear_interpolate_gaussians(self, output_path, t=60, features_to_interpolate=[]):
        """
        Linearly interpolate between matched gaussians and save the sequence as PLY files.
        
        Args:
            output_path: Directory to save the interpolated PLY files
            t: Total number of frames to interpolate (including start and end)
        """
        mkdir_p(output_path)
        device = self.new_gaussians1['xyz'].device

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
                    interpolated[key] = (1 - ratio) * self.new_gaussians1[key] + ratio * self.new_gaussians2[key]
                else:
                    interpolated[key] = self.new_gaussians1[key]
            # SLERP for rotation quaternions
            if 'rotation' in features_to_interpolate:
                interpolated['rotation'] = slerp(
                    self.new_gaussians1['rotation'],
                    self.new_gaussians2['rotation'],
                    ratio
                )
            else:
                interpolated['rotation'] = self.new_gaussians1['rotation']
            
            # Copy active_sh_degree
            interpolated['active_sh_degree'] = self.new_gaussians1['active_sh_degree']
            
            # Save the interpolated gaussians
            frame_path = os.path.join(output_path, f'frame_{i:04d}.ply')
            self.save_new_gaussians(interpolated, frame_path, is_interpolation=True)

    def random_sample_matched_gaussians(self, prec=0.01):
        """
        Randomly sample a percentage of matched point pairs.
        
        Args:
            prec: Percentage of matched pairs to keep (between 0 and 1)
        """
        device = self.new_gaussians1['xyz'].device
        
        # Get total number of matched pairs
        n_matches = len(self.new_matches)
        n_samples = int(n_matches * prec)


        # check if new_matches's elements in list containss first element of tuple and second element of tuple are unique
        first_indices = [m[0] for m in self.new_matches]
        second_indices = [m[1] for m in self.new_matches]
        len_first_indices = len(first_indices)
        len_unique_first_indices = len(set(first_indices))
        len_second_indices = len(second_indices)
        len_unique_second_indices = len(set(second_indices))

        if len_unique_first_indices != len_first_indices:
            raise ValueError("Duplicate first indices found in new_matches")

        if len_unique_second_indices != len_second_indices:
            raise ValueError("Duplicate second indices found in new_matches")
        
        # Randomly select matches
        match_indices = torch.randperm(n_matches)[:n_samples]
        selected_matches = [self.new_matches[i] for i in match_indices]
        
        # Get unique indices for both gaussian sets
        indices1 = sorted(list(set(int(idx1) for idx1, _ in selected_matches)))
        indices2 = sorted(list(set(int(idx2) for _, idx2 in selected_matches)))
        
        # Convert to tensors
        indices1 = torch.tensor(indices1, device=device)
        indices2 = torch.tensor(indices2, device=device)
        
        # Create index mapping for updating matches
        idx_map1 = {int(old_idx.cpu().item()): new_idx for new_idx, old_idx in enumerate(indices1)}
        idx_map2 = {int(old_idx.cpu().item()): new_idx for new_idx, old_idx in enumerate(indices2)}
        
        # Sample gaussians1
        sampled_gaussians1 = {
            key: value[indices1] for key, value in self.new_gaussians1.items() 
            if isinstance(value, torch.Tensor)
        }
        sampled_gaussians1['active_sh_degree'] = self.new_gaussians1['active_sh_degree']
        
        # Sample gaussians2
        sampled_gaussians2 = {
            key: value[indices2] for key, value in self.new_gaussians2.items()
            if isinstance(value, torch.Tensor)
        }
        sampled_gaussians2['active_sh_degree'] = self.new_gaussians2['active_sh_degree']
        
        # Update matches with new indices
        sampled_matches = [(idx_map1[int(idx1)], idx_map2[int(idx2)]) for idx1, idx2 in selected_matches]
        
        # Update class attributes
        self.new_gaussians1 = sampled_gaussians1
        self.new_gaussians2 = sampled_gaussians2
        self.new_matches = sampled_matches
        
        print(f"Randomly sampled {n_samples} matched pairs ({prec*100:.1f}% of original pairs)")
        print(f"Result: {len(indices1)} points in model 1, {len(indices2)} points in model 2")
        return sampled_gaussians1, sampled_gaussians2, sampled_matches


if __name__ == "__main__":
    # Parse arguments
    args, model_params1, model_params2, pipe_params = KNNMatchingInterpolation.parse_args()
    
    # Extract parameters
    dataset1 = model_params1.extract(args)
    dataset2 = model_params2.extract(args)
    pipe = pipe_params.extract(args)

    matching_features = args.matching_features # list of features to use for matching [str]
    interpolation_features = args.interpolation_features # list of features to use for interpolation [str]

    # Create KNN matching instance
    print("Creating KNN matching instance. Loading gaussians...")
    matcher = KNNMatchingInterpolation(dataset1, dataset2, pipe, args.iteration_1, args.iteration_2, replace_2_from_1_part=False)

    if "sdf" in matching_features:
        matcher.neus_model_ckpt_1 = args.neus_model_ckpt_1
        matcher.neus_config_path_1 = args.neus_config_path_1
        matcher.neus_model_ckpt_2 = args.neus_model_ckpt_2
        matcher.neus_config_path_2 = args.neus_config_path_2

    print("Computing bidirectional KNN... Getting initial hard matches...")
    matches_1to2, matches_2to1, nearest_neighbors_1to1, nearest_neighbors_2to2, features1, features2, xyz_1, xyz_2 = matcher.compute_bidirectional_knn(k=1, features_to_use=matching_features)
    print("Refine matches using other loss terms (e.g., local closeness)...")
    matcher.release_gaussians()
    # matches_1to2 = matcher.soft_matching_optimization_minibatch(features1, features2, xyz_2, matches_1to2, nearest_neighbors_1to1)
    # with open("matches_1to2.pkl", "wb") as f:
    #     pickle.dump(matches_1to2, f)
    # matches_2to1 = matcher.soft_matching_optimization_minibatch(features2, features1, xyz_1, matches_2to1, nearest_neighbors_2to2, 2)
    # with open("matches_2to1.pkl", "wb") as f:
    #     pickle.dump(matches_2to1, f)

    # load matches_1to2 and matches_2to1 from files
    # with open("matches_1to2.pkl", "rb") as f:
    #     matches_1to2 = pickle.load(f)
    # with open("matches_2to1.pkl", "rb") as f:
    #     matches_2to1 = pickle.load(f)
    # matches_1to2 = matches_1to2.reshape(-1, 1)
    # matches_2to1 = matches_2to1.reshape(-1, 1)
    
    print("Creating matched gaussians...")
    matcher = KNNMatchingInterpolation(dataset1, dataset2, pipe, args.iteration_1, args.iteration_2, replace_2_from_1_part=False)
    matcher.create_matched_gaussians(matches_1to2, matches_2to1)
    print("Exporting new gaussians...")
    path_new_gaussian1 = os.path.join(dataset1.model_path,
                                "point_cloud",
                                "iteration_" + str(args.iteration_1),
                                "new_point_cloud.ply")
    path_new_gaussian2 = os.path.join(dataset2.model_path,
                                "point_cloud",
                                "iteration_" + str(args.iteration_2),
                                "new_point_cloud.ply")
    matcher.export_new_gaussians(path_new_gaussian1, path_new_gaussian2)
    # print("Randomly sampling matched gaussians...")
    # matcher.random_sample_matched_gaussians(prec=0.01)
    print("Interpolating gaussians...")
    output_path_dir = os.path.join(dataset1.model_path, "point_cloud", "interpolation")
    matcher.linear_interpolate_gaussians(output_path_dir, t=120, features_to_interpolate=interpolation_features)


