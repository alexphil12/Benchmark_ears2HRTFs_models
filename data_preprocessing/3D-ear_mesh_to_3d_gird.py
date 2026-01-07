

"""
This script processes 3D ear mesh files in STL format, calculates their bounding box dimensions, 
and generates a 3D grid representation of the meshes. The resulting grids are saved as NumPy arrays.

Modules:
    - os: For file and directory operations.
    - trimesh: For loading and processing 3D mesh files.
    - numpy: For numerical operations and array manipulations.
    - tqdm: For displaying progress bars during loops.

Constants:
    - path_data: Path to the directory containing the 3D ear mesh files in STL format.
    - path_data2: Path to the directory where the generated 3D grid data will be saved.

Workflow:
    1. Load all STL files from the specified directory.
    2. Calculate the global minimum and maximum bounds (x, y, z) across all meshes.
    3. Generate a 3D grid with a specified resolution (step size of 5 units).
    4. For each mesh:
        a. Create a 3D grid representation where each voxel indicates whether it is inside the mesh.
        b. Save the grid as a NumPy array file.

Functions:
    - trimesh.load_mesh: Loads a 3D mesh from a file.
    - trimesh.bounds: Retrieves the bounding box of the mesh.
    - trimesh.ray.contains_points: Checks if a point is inside the mesh.

Output:
    - For each mesh, a NumPy array file is saved in the specified output directory. 
      The file name corresponds to the original mesh file name with "_ear_grid.npy" appended.

Notes:
    - The grid resolution is determined by the step size (5 units in this case).
    - Ensure that the input and output directories exist and contain the appropriate files.
"""
import os
import trimesh
import numpy as np 
from tqdm import trange

path_data="/home/PhilipponA/HRTF_sonicom_dataset/Mesh_3D_ears/STL_mesh/"
path_data2="/home/PhilipponA/HRTF_sonicom_dataset/Mesh_3D_ears/Grid_sampled_data/"
list_subject=os.listdir(path_data)
list_subject.sort()
for j in trange(len(list_subject),desc="Calculating max min bounds of the meshs"):
    mesh = trimesh.load_mesh(os.path.join(path_data,list_subject[j]))
    if j==0:
        x_min, y_min, z_min = mesh.bounds[0]
        x_max, y_max, z_max = mesh.bounds[1]
    else:
        x_min = min(x_min, mesh.bounds[0][0])
        y_min = min(y_min, mesh.bounds[0][1])
        z_min = min(z_min, mesh.bounds[0][2])
        x_max = max(x_max, mesh.bounds[1][0])
        y_max = max(y_max, mesh.bounds[1][1])
        z_max = max(z_max, mesh.bounds[1][2])
Precission=3
x_vals = np.arange(x_min, x_max, Precission)
y_vals = np.arange(y_min, y_max, Precission)
z_vals = np.arange(z_min, z_max, Precission)

for l in trange(len(list_subject),desc="Calculating the grid of the meshs"):
    mesh = trimesh.load_mesh(os.path.join(path_data,list_subject[l]))
    grid=np.zeros((len(x_vals),len(y_vals),len(z_vals)), dtype=np.int32)
    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            for k, z in enumerate(z_vals):
                # Vérifier si le point (x, y, z) est dans l'objet
                point = np.array([x, y, z])
                point=np.reshape(point,(1,3))
                bool_val=mesh.ray.contains_points(point)[0]
                if bool_val==np.True_ :
                    grid[i, j, k] = 1  # L'objet est présent à cette position
    np.save(os.path.join(path_data2,list_subject[l][:-4]+"_precission_factor="+str(Precission)+"_ear_grid.npy"),grid)
