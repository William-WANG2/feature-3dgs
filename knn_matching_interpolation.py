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

class KNNMatchingInterpolation:
    def __init__(self, dataset1, dataset2, pipe, iteration1=None, iteration2=None):
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

    def compute_bidirectional_knn(self, k=1):
        """
        Compute bidirectional KNN matching between two GaussianModels using semantic features
        Args:
            k: Number of nearest neighbors (default=1)
        Returns:
            matches_1to2: Indices of nearest neighbors in gaussians2 for each point in gaussians1 dim: (N1, k) format in numpy
            matches_2to1: Indices of nearest neighbors in gaussians1 for each point in gaussians2 dim: (N2, k) format in numpy
        """
        # Get semantic features from both models and reshape them
        features1 = self.gaussians1.get_semantic_feature.detach().squeeze(-1).cpu().numpy()  # (N1, semantic_feature_size)
        features2 = self.gaussians2.get_semantic_feature.detach().squeeze(-1).cpu().numpy()  # (N2, semantic_feature_size)
        features1 = features1.squeeze(1)
        features2 = features2.squeeze(1)
        # Ensure features are float32 (required by faiss)
        features1 = features1.astype(np.float32)
        features2 = features2.astype(np.float32)

        # Create faiss index for fast KNN
        d = features1.shape[1]  # feature dimension
        
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

        self.new_gaussians1 = new_gaussians1
        self.new_gaussians2 = new_gaussians2
        self.new_matches = new_matches

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

    def save_new_gaussians(self, gaussians, path_gaussian):
        mkdir_p(os.path.dirname(path_gaussian))
        xyz = gaussians['xyz'].detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = gaussians['features_dc'].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = gaussians['features_rest'].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = gaussians['opacity'].detach().cpu().numpy()
        scale = gaussians['scaling'].detach().cpu().numpy()
        rotation = gaussians['rotation'].detach().cpu().numpy()
        semantic_feature = gaussians['semantic_feature'].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, semantic_feature), axis=1) 
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path_gaussian)
    

    def export_new_gaussians(self, path_gaussian1, path_gaussian2):
        self.save_new_gaussians(self.new_gaussians1, path_gaussian1)
        self.save_new_gaussians(self.new_gaussians2, path_gaussian2)

    def interpolate_gaussians(self, output_path, t=60):
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
            for key in ['xyz', 'features_dc', 'features_rest', 'scaling', 'opacity', 'semantic_feature']:
                if key == 'semantic_feature':
                    interpolated[key] = self.new_gaussians1[key]
                else:
                    interpolated[key] = (1 - ratio) * self.new_gaussians1[key] + ratio * self.new_gaussians2[key]
                

            
            # SLERP for rotation quaternions
            interpolated['rotation'] = slerp(
                self.new_gaussians1['rotation'],
                self.new_gaussians2['rotation'],
                ratio
            )
            
            # Copy active_sh_degree
            interpolated['active_sh_degree'] = self.new_gaussians1['active_sh_degree']
            
            # Save the interpolated gaussians
            frame_path = os.path.join(output_path, f'frame_{i:04d}.ply')
            self.save_new_gaussians(interpolated, frame_path)

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
    
    # Create KNN matching instance
    print("Creating KNN matching instance. Loading gaussians...")
    matcher = KNNMatchingInterpolation(dataset1, dataset2, pipe, args.iteration_1, args.iteration_2)
    print("Computing bidirectional KNN... Getting matches...")
    matcher.compute_bidirectional_knn(k=1)
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
    # print("Interpolating gaussians...")
    output_path_dir = os.path.join(dataset1.model_path, "point_cloud", "interpolation")
    matcher.interpolate_gaussians(output_path_dir, t=60)


