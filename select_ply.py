import argparse
import random
import numpy as np
from plyfile import PlyData, PlyElement

def read_ply(file_path):
    """
    Reads a PLY file and returns the PlyData object.
    """
    with open(file_path, 'rb') as f:
        ply_data = PlyData.read(f)
    return ply_data

def write_ply(file_path, vertices, faces):
    """
    Writes vertices and faces to a PLY file.
    """
    vertex_element = PlyElement.describe(vertices, 'vertex')
    if faces is not None:
        face_element = PlyElement.describe(faces, 'face')
        PlyData([vertex_element, face_element], text=True).write(file_path)
    else:
        PlyData([vertex_element], text=True).write(file_path)

def select_random_vertices(ply_data, portion):
    """
    Select a random portion of vertices from the PLY data.
    """
    vertices = ply_data['vertex']
    num_vertices = len(vertices)
    num_selected = int(num_vertices * portion)

    if num_selected == 0:
        raise ValueError("The specified portion is too small to select any vertices.")

    selected_indices = random.sample(range(num_vertices), num_selected)
    selected_vertices = vertices[selected_indices]

    if 'face' in ply_data:
        faces = ply_data['face']
        new_faces = []
        selected_indices_set = set(selected_indices)

        for face in faces:
            vertex_indices = face['vertex_indices'].tolist()
            if all(idx in selected_indices_set for idx in vertex_indices):
                new_faces.append((vertex_indices,))

        new_faces = np.array(new_faces, dtype=[('vertex_indices', 'O')]) if new_faces else None
    else:
        new_faces = None

    return selected_vertices, new_faces

def main():
    parser = argparse.ArgumentParser(description='Randomly select a portion of vertices from a PLY file.')
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input PLY file.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output PLY file.')
    parser.add_argument('--portion', type=float, required=True, help='Portion of vertices to select (0.0 to 1.0).')

    args = parser.parse_args()

    if not (0.0 < args.portion <= 1.0):
        raise ValueError("The portion must be between 0.0 and 1.0 (exclusive for 0.0).")

    ply_data = read_ply(args.input_path)
    selected_vertices, new_faces = select_random_vertices(ply_data, args.portion)

    write_ply(args.output_path, selected_vertices, new_faces)

    print(f"Successfully written a subset with {len(selected_vertices)} vertices to {args.output_path}")

if __name__ == '__main__':
    main()
