import trimesh
"""
This script processes 3D mesh files to extract ear regions from a dataset of 3D scans. 
It uses curvature analysis and clustering to identify and isolate the ear regions 
from the mesh, and saves the extracted ear meshes as STL files.
Modules:
    - trimesh: For handling 3D mesh files and performing geometric computations.
    - numpy: For numerical operations and array manipulations.
    - sklearn.cluster.DBSCAN: For clustering high-curvature points to identify ear regions.
    - os: For file and directory operations.
    - tqdm: For progress visualization during loops.
Constants:
    - path_data: Path to the dataset containing 3D mesh files.
Workflow:
    1. Load the list of subjects from the dataset directory.
    2. For each subject:
        a. Attempt to load the preprocessed watertight mesh file. If unavailable, load the raw watertight mesh file.
        b. Compute the discrete Gaussian curvature of the mesh vertices.
        c. Normalize the curvature values and select vertices with high curvature (potential ear regions).
        d. Use DBSCAN clustering to group high-curvature points into clusters.
        e. Identify the two largest clusters, corresponding to the left and right ears.
        f. Extract the vertices and faces corresponding to each ear cluster.
        g. Save the extracted ear meshes as STL files.
Functions:
    - trimesh.curvature.discrete_gaussian_curvature_measure: Computes curvature for mesh vertices.
    - DBSCAN.fit: Clusters high-curvature points.
    - trimesh.Trimesh: Creates a new mesh object for the extracted ear regions.
    - trimesh.Trimesh.export: Exports the extracted ear mesh to an STL file.
Notes:
    - The script assumes a specific directory structure for the dataset.
    - The clustering parameters (eps and min_samples) may need adjustment based on the scale of the mesh.
    - The script handles missing files gracefully by printing a warning and skipping the subject.
"""
import numpy as np
from sklearn.cluster import DBSCAN
import os
from tqdm import tqdm
from tqdm import trange


path_data="/databases/sonicom_hrtf_dataset"
list_subject=os.listdir(path_data)
list_subject.sort()

# Charger le maillage
for u in tqdm(range(1,100),desc="Subjects extracted"):

    try:
        mesh = trimesh.load_mesh("/databases/sonicom_hrtf_dataset/"+list_subject[u]+"/3DSCAN/"+list_subject[u]+"_wtt_prepd.stl")
    except:
        print("The subject "+list_subject[u]+" has no 3DSCAN wtt_predp.stl file")
        mesh = trimesh.load_mesh("/databases/sonicom_hrtf_dataset/"+list_subject[u]+"/3DSCAN/"+list_subject[u]+"_watertight.stl")
        type_mesh="watertight"
    else:
        type_mesh="wtt_prepd"

    # Calculer la courbure des sommets
    curvature = trimesh.curvature.discrete_gaussian_curvature_measure(
    mesh, mesh.vertices, radius=2)
    # Normalize curvature values
    curvature = np.abs(curvature)
    curvature = (curvature - curvature.min())*100 /(curvature.max() - curvature.min())

    # Sélectionner les sommets de forte courbure (potentielles oreilles)
    threshold = np.percentile(curvature, 90)  # Top 10% des points les plus courbés
    high_curvature_indices = np.where(curvature > threshold)[0]

    # Extraire les coordonnées des sommets candidats
    high_curvature_points = mesh.vertices[high_curvature_indices]

    dbscan = DBSCAN(eps=10, min_samples=100).fit(high_curvature_points)  # Adjust eps based on scale
    labels = dbscan.labels_

    # Find the two largest clusters (corresponding to the ears)
    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    # Get top 2 clusters (ear regions)
    largest_clusters = unique_labels[np.argsort(counts)][-2:]
    # %%
    # Extraction du plus grand cluster (oreille 1)
    ear_cluster_label_1 = largest_clusters[0]  # Select one ear
    ear_indices_1 = high_curvature_indices[labels == ear_cluster_label_1]
    ear_values_1=high_curvature_points[labels == ear_cluster_label_1]

    Min_ear_1=np.min(ear_values_1,axis=0)
    Max_ear_1=np.max(ear_values_1,axis=0)
    ear_indices_1=np.where((mesh.vertices[:,0]>=Min_ear_1[0])&(mesh.vertices[:,0]<= Max_ear_1[0])&(mesh.vertices[:,1]>=Min_ear_1[1])&(mesh.vertices[:,1]<= Max_ear_1[1])&(mesh.vertices[:,2]>=Min_ear_1[2])&(mesh.vertices[:,2]<= Max_ear_1[2]))
    ear_indices_1=ear_indices_1[0]
    ear_faces_1 = []
    faces=mesh.faces
    for face in tqdm(faces,desc="Extracting faces of the ear 1 for the subject "+list_subject[u]):
        L = 0
        for j in range(0, 3):
            if face[j] not in ear_indices_1:
                L = 1
        if L == 0:
            face_index = [0, 0, 0]
            for l in range(0, 3):
                face_index[l] = list(ear_indices_1).index(face[l])         
            ear_faces_1.append(face_index)
    ear_faces_1 = np.array(ear_faces_1)
    verti = mesh.vertices[:, :]
    ear_mesh_1 = trimesh.Trimesh(vertices=verti[ear_indices_1], faces=ear_faces_1)

    # Save the extracted ear as an STL file
    ear_mesh_1.export("/home/PhilipponA/HRTF_sonicom_dataset/Mesh_3D_ears/STL_mesh/"+ list_subject[u] +"_" +type_mesh +"_ear_extracted_1.stl")
    
    # Extraction du 2ème plus grand cluster (oreille 2)
    ear_cluster_label_2 = largest_clusters[1]  # Select one ear
    ear_indices_2 = high_curvature_indices[labels == ear_cluster_label_2]
    ear_values_2 = high_curvature_points[labels == ear_cluster_label_2]
    
    Min_ear_2 = np.min(ear_values_2, axis=0)
    Max_ear_2 = np.max(ear_values_2, axis=0)
    ear_indices_2 = np.where((mesh.vertices[:, 0] >= Min_ear_2[0]) & (mesh.vertices[:, 0] <= Max_ear_2[0]) & 
                                 (mesh.vertices[:, 1] >= Min_ear_2[1]) & (mesh.vertices[:, 1] <= Max_ear_2[1]) & 
                                 (mesh.vertices[:, 2] >= Min_ear_2[2]) & (mesh.vertices[:, 2] <= Max_ear_2[2]))
    ear_indices_2 = ear_indices_2[0]
    ear_faces_2 = []
    faces = mesh.faces
    for face in tqdm(faces,desc="Extracting faces of the ear 2 for the subject "+list_subject[u]):
        L = 0
        for j in range(0, 3):
            if face[j] not in ear_indices_2:
                L = 1
        if L == 0:
            face_index = [0, 0, 0]
            for l in range(0, 3):
                face_index[l] = list(ear_indices_2).index(face[l])
            ear_faces_2.append(face_index)
    ear_faces_2 = np.array(ear_faces_2)
    verti = mesh.vertices[:, :]
    ear_mesh_2 = trimesh.Trimesh(vertices=verti[ear_indices_2], faces=ear_faces_2)
    
        # Save the extracted ear as an STL file
    ear_mesh_2.export("/home/PhilipponA/HRTF_sonicom_dataset/Mesh_3D_ears/STL_mesh/"+ list_subject[u]+"_" +type_mesh +"_ear_extracted_2.stl")
