import os
import sys
from scene import Scene, GaussianModel
from argparse import ArgumentParser
from arguments import ModelParamsMatching, PipelineParams
import torch
import faiss
import numpy as np
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement

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

    def compute_bidirectional_knn(self, k=1, features_to_use=['semantic_feature']):
        """
        Compute bidirectional KNN matching between two GaussianModels using multiple features
        Args:
            k: Number of nearest neighbors (default=1)
            features_to_use: List of feature names to use for matching (e.g., ['semantic_feature', 'xyz'])
        Returns:
            matches_1to2: Indices of nearest neighbors in gaussians2 for each point in gaussians1 dim: (N1, k)
            matches_2to1: Indices of nearest neighbors in gaussians1 for each point in gaussians2 dim: (N2, k)
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

        # store the matches in numpy array
        self.matches_1to2 = matches_1to2
        self.matches_2to1 = matches_2to1

        # # Convert to torch tensors and move to GPU if available
        # matches_1to2 = torch.from_numpy(matches_1to2).squeeze(-1)
        # matches_2to1 = torch.from_numpy(matches_2to1).squeeze(-1)
        
        # if torch.cuda.is_available():
        #     matches_1to2 = matches_1to2.cuda()
        #     matches_2to1 = matches_2to1.cuda()

        return matches_1to2, matches_2to1
    
    def soft_matching(self, k=1, features_to_use=['semantic_feature']):
        """
        Compute soft matching between two GaussianModels using multiple features
        """

    def create_matched_gaussians(self):
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
        for i in range(self.matches_1to2.shape[0]):
            k_values_1[i] += 1
            k_values_2[self.matches_1to2[i, 0]] += 1
            
        # Count from matches_2to1
        for i in range(self.matches_2to1.shape[0]):
            k_values_2[i] += 1
            k_values_1[self.matches_2to1[i, 0]] += 1
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
        for i in range(self.matches_1to2.shape[0]):
            if k_values_1[i] > 0:  # If gaussian has any links
                # Add k copies of gaussian1[i]
                k = int(k_values_1[i])
                for _ in range(k):
                    indices_1.append(i)
                
                # Add match to matches_1to2[i]
                j = self.matches_1to2[i, 0]
                new_matches.append((current_idx_1[i], current_idx_2[j]))
                current_idx_1[i] += 1
                current_idx_2[j] += 1
        
        # Process matches_2to1
        for i in range(self.matches_2to1.shape[0]):
            if k_values_2[i] > 0:  # If gaussian has any links
                # Add k copies of gaussian2[i]
                k = int(k_values_2[i])
                for _ in range(k):
                    indices_2.append(i)
                
                # Add match to matches_2to1[i]
                j = self.matches_2to1[i, 0]
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


        # check if new_matches's elements in list contains first element of tuple and second element of tuple are unique
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

    print("Computing bidirectional KNN... Getting matches...")
    matcher.compute_bidirectional_knn(k=1, features_to_use=matching_features)
    print("Creating matched gaussians...")
    matcher.create_matched_gaussians()
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


