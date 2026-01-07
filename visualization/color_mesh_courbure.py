import trimesh
import numpy as np
import matplotlib.pyplot as pyplot
import os
def spread_data(X):
    X=(X-np.min(X))/(np.max(X)-np.min(X))
    X=np.log(X+1e-10)
    X=(X-np.min(X))/(np.max(X)-np.min(X))
    return X

def colorer_et_sauvegarder_maillage_manuel(input_stl_path, output_ply_path):
    """
    Charge, colore le maillage et écrit manuellement le contenu du fichier PLY
    pour garantir l'inclusion des couleurs sans dépendre des fonctions internes d'export.
    """
    try:
        print(f"Chargement du maillage depuis: {input_stl_path}...")
        mesh = trimesh.load_mesh(input_stl_path)
    except Exception as e:
        print(f"Erreur lors du chargement: {e}")
        return

    if not isinstance(mesh, trimesh.Trimesh):
        if isinstance(mesh, list):
            mesh = trimesh.util.concatenate(mesh)
        if not isinstance(mesh, trimesh.Trimesh):
            print("Erreur : Le fichier n'est pas un maillage triangulaire valide.")
            return

    # --- 1. Calcul de la Courbure et Coloration (inchangé) ---
    print("Calcul de la courbure gaussienne...")
    try:
        gaussian_curvature = trimesh.curvature.discrete_gaussian_curvature_measure(
    mesh, mesh.vertices, radius=2)
    except ValueError as e:
        print(f"Erreur lors du calcul de courbure : {e}")
        return

    abs_curvature = np.abs(gaussian_curvature)
    max_curv = np.max(abs_curvature)

    
    if max_curv < np.finfo(float).eps:
        normalized_curvature = np.zeros_like(abs_curvature)
    else:
        normalized_curvature = abs_curvature / max_curv
    pyplot.figure()
    for j in range (1):
        normalized_curvature = spread_data(normalized_curvature)
    pyplot.hist(normalized_curvature, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    pyplot.title('Histogramme de la courbure normalisée')
    pyplot.savefig('histogram_curvature.png')

    cmap = pyplot.get_cmap('inferno') 
    rgba_colors = cmap(normalized_curvature)
    
    # Couleurs RGB (0-255) entières
    vertex_colors = (rgba_colors[:, :3] * 255).astype(np.uint8)
    
    # --- 2. Préparation des Données du Fichier PLY ---
    
    # Concaténation des coordonnées des sommets (XYZ) et des couleurs (RGB)
    # Les coordonnées sont généralement en float32, les couleurs en ubyte (uint8)
    vertex_data = np.hstack((
        mesh.vertices.astype(np.float32), 
        vertex_colors
    ))
    
    # Les faces dans PLY sont des listes d'indices (0-based) précédées du nombre d'indices (généralement 3 pour les triangles)
    # Ex: [3, i1, i2, i3]
    face_data = np.hstack((
        np.full((len(mesh.faces), 1), 3, dtype=np.int32),  # Ajoute le '3' au début de chaque ligne
        mesh.faces
    ))
    
    num_vertices = len(mesh.vertices)
    num_faces = len(mesh.faces)
    
    # --- 3. Construction du Fichier PLY ---
    print(f"Génération du contenu PLY (ASCII) pour {num_vertices} sommets et {num_faces} faces...")

    header = f"""ply
format ascii 1.0
element vertex {num_vertices}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
element face {num_faces}
property list uchar int vertex_indices
end_header
"""
    # Écriture du fichier
    try:
        with open(output_ply_path, 'w') as f:
            f.write(header)
            
            # Écriture des données des sommets (X Y Z R G B)
            np.savetxt(f, vertex_data, fmt='%.6f %s %s %d %d %d') 
            
            # Écriture des données des faces (3 i1 i2 i3)
            # %d pour les entiers
            np.savetxt(f, face_data, fmt='%d %d %d %d')

        print("\n✅ Opération terminée (Écriture Manuelle PLY) !")
        print(f"Le maillage coloré a été sauvegardé dans : {os.path.abspath(output_ply_path)}")
        print("Ceci est la méthode la plus compatible, assurez-vous de vérifier le mode d'affichage des couleurs dans votre visualiseur.")

    except Exception as e:
        print(f"\n❌ ERREUR lors de l'écriture manuelle du fichier PLY. Erreur : {e}")

# --- Bloc de test (inchangé) ---
INPUT_FILE = 'P0107_wtt_prepd.stl' 
OUTPUT_FILE = 'mesh_colored.ply'

if not os.path.exists(INPUT_FILE):
    try:
        print("[INFO]: Création d'un tore de test pour l'exemple...")
        test_mesh = trimesh.creation.torus()
        test_mesh.export(INPUT_FILE)
    except:
        pass 

# Appel de la fonction corrigée
colorer_et_sauvegarder_maillage_manuel(INPUT_FILE, OUTPUT_FILE)