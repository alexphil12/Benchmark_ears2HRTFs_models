
import os
import numpy as np
import cv2
from PIL import Image
import tqdm
import matplotlib.pyplot as plt

Path_3D="/home/PhilipponA/HRTF_sonicom_dataset/Mesh_3D_ears/STL_mesh"

path_ear_photo="/home/PhilipponA/HRTF_sonicom_dataset/Photo_2D_ears/PNG_ear_photo/"

path_ear_photo_no_disto="/home/PhilipponA/HRTF_sonicom_dataset/Photo_2D_ears/PNG_ear_photo_no_disto/"

list_subject=os.listdir(Path_3D)
list_subject.sort()
subject_name=[]
for u in range(len(list_subject)//2):
    subject_name.append(list_subject[2*u][0:5])

max_height=0
max_width=0
for j in tqdm.trange(len(subject_name),desc="find max dimension of photo"):
    ima=cv2.imread(os.path.join(path_ear_photo,subject_name[j]+"_left_ear.png"))
    if ima.shape[0]>max_height:
        max_height=ima.shape[0]
    if ima.shape[1]>max_width:
        max_width=ima.shape[1]
    ima2=cv2.imread(os.path.join(path_ear_photo,subject_name[j]+"_right_ear.png"))
    if ima2.shape[0]>max_height:
        max_height=ima2.shape[0]
    if ima2.shape[1]>max_width:
        max_width=ima2.shape[1]
for j in tqdm.trange(len(subject_name),desc="Resize photo to max dimension"):
    ima=cv2.imread(os.path.join(path_ear_photo,subject_name[j]+"_left_ear.png"))
    ima2=cv2.imread(os.path.join(path_ear_photo,subject_name[j]+"_right_ear.png"))
    New_ima_left=np.ones((max_height,max_width,3),dtype=np.uint8)*0
    New_ima_right=np.ones((max_height,max_width,3),dtype=np.uint8)*0
    New_ima_left[0:ima.shape[0],0:ima.shape[1],:]=ima
    New_ima_right[0:ima2.shape[0],0:ima2.shape[1],:]=ima2
    cv2.imwrite(os.path.join(path_ear_photo_no_disto,subject_name[j]+"_left_ear.png"),New_ima_left)
    cv2.imwrite(os.path.join(path_ear_photo_no_disto,subject_name[j]+"_right_ear.png"),New_ima_right)

