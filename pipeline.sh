#!/bin/zsh

# List of names
# names=("car1" "car2")
names=("cat" "dog")

# Loop over each name in the list
for name in "${names[@]}"
do
  echo "Processing $name..."

#   # Run export_image_embeddings.py
#   python ./encoders/dino_encoder/export_image_embeddings.py --model_name dinov2_vitl14 --input "data/nerf_synthetic/${name}/train" --output "data/nerf_synthetic/${name}/dino_embeddings/train/"

  # Run train.py
  python train.py -s "data/nerf_synthetic/${name}" -m "output/nerf_synthetic/${name}" -f dinov2 --iterations 7000 --test_iterations 7000

  echo "$name processing completed."
done

echo "All tasks completed."
