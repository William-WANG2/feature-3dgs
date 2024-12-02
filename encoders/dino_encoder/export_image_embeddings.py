import argparse
import os
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
import cv2
from sklearn.decomposition import PCA

class Dinov2Matcher:
    def __init__(
        self,
        model_name,
        smaller_edge_size=448,
        half_precision=False,
        device="cuda"
    ):
        """
        Initializes the Dinov2Matcher class by loading the DINOv2 model
        from the specified model_name and setting up the image transformation pipeline.

        Parameters:
            model_name (str): The path to the DINOv2 model_name.
            smaller_edge_size (int): The size to which the smaller edge of the image is resized.
            half_precision (bool): Whether to use half-precision (FP16) for the model.
            device (str): The device to run the model on (e.g., "cuda" or "cpu").
        """
        self.smaller_edge_size = smaller_edge_size
        self.half_precision = half_precision
        self.device = device

        # Load the pre-trained DINOv2 model from the specified model_name
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model.to(self.device)
        if self.half_precision:
            self.model.half()

        self.model.eval()  # Set the model to evaluation mode

        # Define the image transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize(
                size=smaller_edge_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True
            ),
            transforms.ToTensor(),  # Convert image to tensor
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),  # Normalize with ImageNet mean
                std=(0.229, 0.224, 0.225)    # Normalize with ImageNet std
            ),
        ])

    def prepare_image(self, rgb_image_numpy):
        """
        Prepares the input image for feature extraction by resizing, normalizing,
        and cropping it to dimensions compatible with the model's patch size.

        Parameters:
            rgb_image_numpy (np.ndarray): Input RGB image as a NumPy array.

        Returns:
            image_tensor (torch.Tensor): The transformed image tensor.
            grid_size (tuple): The size of the grid (height, width) after processing.
            resize_scale (float): The scale factor used during resizing.
        """
        image = Image.fromarray(rgb_image_numpy)  # Convert NumPy array to PIL Image
        image_tensor = self.transform(image)      # Apply transformations
        resize_scale = image.width / image_tensor.shape[2]  # Calculate the scale factor

        # Crop image dimensions to be multiples of the patch size
        height, width = image_tensor.shape[1:]  # Get height and width from tensor
        cropped_width = width - width % self.model.patch_size
        cropped_height = height - height % self.model.patch_size
        # Crop the image tensor
        image_tensor = image_tensor[:, :cropped_height, :cropped_width]

        # Calculate the grid size based on the patch size
        grid_size = (
            cropped_height // self.model.patch_size,
            cropped_width // self.model.patch_size
        )
        return image_tensor, grid_size, resize_scale

    def extract_features(self, image_tensor):
        """
        Extracts feature embeddings (tokens) from the image using the DINOv2 model.

        Parameters:
            image_tensor (torch.Tensor): The preprocessed image tensor.

        Returns:
            tokens (np.ndarray): The extracted feature tokens as a NumPy array. Shape is (C, H, W).
        """
        with torch.inference_mode():
            # Prepare the image batch
            if self.half_precision:
                image_batch = image_tensor.unsqueeze(0).half().to(self.device)
            else:
                image_batch = image_tensor.unsqueeze(0).to(self.device)

            # Get intermediate layers (tokens) from the model
            tokens = self.model.get_intermediate_layers(image_batch)[0].squeeze()
            image_height = image_tensor.shape[1]  # Get the height of the image
            image_width = image_tensor.shape[2]  # Get the width of the image
            H = image_height // self.model.patch_size  # Calculate the grid height
            W = image_width // self.model.patch_size    # Calculate the grid width
            tokens = tokens.view(-1, H, W)  # Reshape tokens to C*H*W
        return tokens.cpu().numpy()  # Convert tokens to NumPy array
    def get_embedding_visualization(self, tokens, grid_size):
        """
        Visualizes the embeddings by reducing their dimensions using PCA and normalizing
        the result for display.

        Parameters:
            tokens (np.ndarray): The feature tokens extracted from the image.
            grid_size (tuple): The grid size (height, width) from the processed image.

        Returns:
            normalized_tokens (np.ndarray): The embeddings reshaped and normalized for visualization.
        """
        pca = PCA(n_components=3)  # Initialize PCA for 3 components
        flattened_tokens = tokens.reshape(-1, tokens.shape[0])  # Flatten tokens to (N, C)
        print("flattened_tokens shape: ", flattened_tokens.shape)
        reduced_tokens = pca.fit_transform(flattened_tokens)  # Apply PCA
        print("reduced_tokens shape: ", reduced_tokens.shape)

        # Reshape the reduced tokens to match the grid size
        reduced_tokens = reduced_tokens.reshape((*grid_size, -1))
        # Normalize the tokens for visualization
        normalized_tokens = (reduced_tokens - np.min(reduced_tokens)) / (np.max(reduced_tokens) - np.min(reduced_tokens))
        return normalized_tokens

# Parser arguments
parser = argparse.ArgumentParser(
    description=(
        "Get image embeddings of an input image or directory of images."
    )
)

parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Path to either a single input image or folder of images.",
)

parser.add_argument(
    "--output",
    type=str,
    required=True,
    help=(
        "Path to the directory where embeddings will be saved. Output will be either a folder "
        "of .pt per image or a single .pt representing image embeddings."
    ),
)

parser.add_argument(
    "--model_name",
    type=str,
    required=True,
    help="The DINOv2 pre-trained model name",
)

parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")

args = parser.parse_args()

# Validate input path
if os.path.isfile(args.input):
    # Single image
    image_paths = [args.input]
    is_single_image = True
elif os.path.isdir(args.input):
    # Directory of images
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_paths = [
        os.path.join(args.input, fname)
        for fname in os.listdir(args.input)
        if os.path.splitext(fname.lower())[1] in valid_extensions and 'depth' not in fname.lower() # filter out depth images for Nerf Synthetic dataset
    ]
    is_single_image = False
else:
    raise ValueError("The input path must be a file or directory.")

# Create output directory if it doesn't exist
os.makedirs(args.output, exist_ok=True)

# Initialize the Dinov2Matcher
dm = Dinov2Matcher(
    model_name=args.model_name,
    half_precision=False,
    device=args.device,
    smaller_edge_size=798
)

# Process images
embeddings_list = []

for image_path in image_paths:
    # Load the image
    image = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    # Extract features from the image
    image_tensor, grid_size, resize_scale = dm.prepare_image(image)
    features = dm.extract_features(image_tensor)
    # # visualization
    # pca_feature =  dm.get_embedding_visualization(features, grid_size)
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,20))
    # ax1.imshow(image)
    # ax2.imshow(pca_feature)
    # fig.tight_layout()
    # plt.show()
    print(f"Extracted features from {image_path} with shape: {features.shape}")

    if not is_single_image:
        # Save each embedding separately
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(args.output, f"{img_name}_fmap_CxHxW.pt")
        torch.save(torch.tensor(features), output_path)
        print(f"Saved features to {output_path}")

if is_single_image:
    # Save the single embedding
    img_name = os.path.splitext(os.path.basename(args.input))[0]
    output_path = os.path.join(args.output, f"{img_name}_fmap_CxHxW.pt")
    torch.save(embeddings_list[0], output_path)
