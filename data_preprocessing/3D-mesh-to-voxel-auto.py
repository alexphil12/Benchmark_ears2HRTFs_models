import os
import trimesh
import numpy as np 
from tqdm import trange
from trimesh.voxel.creation import voxelize

path_data="/home/PhilipponA/HRTF_sonicom_dataset/Mesh_3D_ears/STL_mesh/"
path_data2="/home/PhilipponA/HRTF_sonicom_dataset/Mesh_3D_ears/Grid_sampled_data/"
list_subject=os.listdir(path_data)
list_subject.sort()
for j in trange(len(list_subject),desc="Calculating the grid of the meshs"):
    grid=np.zeros((85,85,85)) #max dim in the dataset is 85
    mesh = trimesh.load_mesh(os.path.join(path_data,list_subject[j]))
    if 'left' in list_subject[j]:
        angle_degrees = 90
    elif 'right' in list_subject[j]:
        angle_degrees = 0
    angle_radians = np.radians(angle_degrees)
    rotation2 = trimesh.transformations.rotation_matrix(
    angle_radians,  # Angle in radians
    [0, 0, 1],      # Axis of rotation (X=1, Y=2, Z=3)
    point=mesh.centroid  # Rotate around the mesh's center
)
    mesh.apply_transform(rotation2)
    voxeL = voxelize(mesh, pitch=1)
    one=voxeL.matrix
    grid[0:one.shape[0],0:one.shape[1],0:one.shape[2]]=one
    if 'left' in list_subject[j]:      
        np.save(os.path.join(path_data2,list_subject[j][0:5]+"_left_ear_grid.npy"),grid)
    else:
        np.save(os.path.join(path_data2,list_subject[j][0:5]+"_right_ear_grid.npy"),grid)
