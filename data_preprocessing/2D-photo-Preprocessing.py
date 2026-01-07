"""
This script preprocesses 2D photos for ear detection and saves the processed images.

Modules:
    - os: Provides functions for interacting with the operating system.
    - pyheif: Used to read HEIC image files.
    - numpy: Provides support for numerical operations on arrays.
    - cv2: OpenCV library for image processing.
    - PIL (Image): Used for image manipulation.
    - tqdm: Provides progress bars for loops.
    - matplotlib.pyplot: Used for plotting and saving images.
    - matplotlib.image.imsave: Saves images to disk.

Constants:
    - Path_3D: Path to the directory containing 3D ear mesh data.
    - path_raw_photo: Path to the directory containing raw HEIC photos.
    - path_all_photo: Path to save all converted PNG photos.
    - path_ear_photo: Path to save cropped ear photos.

Workflow:
    1. Extract subject names from the 3D ear mesh directory.
    2. Convert HEIC images to PNG format:
        - Reads HEIC images using `pyheif`.
        - Converts them to PNG format using PIL and saves them in the `path_all_photo` directory.
    3. Detect and crop ear regions from PNG images:
        - Loads PNG images using OpenCV.
        - Uses Haar cascade classifiers to detect right and left ears.
        - Crops a 224x224 region around the detected ear.
        - Saves the cropped ear images in the `path_ear_photo` directory.

Functions:
    - None explicitly defined. The script uses inline code for processing.

Dependencies:
    - Haar cascade XML files for ear detection:
        - `haarcascade_mcs_rightear.xml`
        - `haarcascade_mcs_leftear.xml`

Notes:
    - Ensure that the Haar cascade XML files are present at the specified paths.
    - The script assumes a specific directory structure for input and output paths.
    - The cropped ear regions are centered around the middle of the image, assuming the ear is located there.
"""
import os
import pyheif
import numpy as np
import cv2
from PIL import Image
import tqdm
import matplotlib.pyplot as plt



Path_3D="/home/PhilipponA/HRTF_sonicom_dataset/Mesh_3D_ears/STL_mesh"

list_subject=os.listdir(Path_3D)
list_subject.sort()
subject_name=[]
for u in range(len(list_subject)//2):
    subject_name.append(list_subject[2*u][0:5])

path_raw_photo="/home/PhilipponA/raw_photo/SONICOM_HRTF_RGB_DATA/"
path_all_photo="/home/PhilipponA/HRTF_sonicom_dataset/Photo_2D_ears/PNG_all_photo/"
path_ear_photo="/home/PhilipponA/HRTF_sonicom_dataset/Photo_2D_ears/PNG_ear_photo/"
path_ear_resized_photo="/home/PhilipponA/HRTF_sonicom_dataset/Photo_2D_ears/PNG_ear_resized_photo/"
# for j in tqdm.trange(len(subject_name),desc="Convert HEIC to PNG"):
#     list_image=os.listdir(os.path.join(path_raw_photo, subject_name[j],"PHOTOGRAMMETRY"))
#     list_image.sort()
#     image_1= pyheif.read(os.path.join(path_raw_photo, subject_name[j],"PHOTOGRAMMETRY", list_image[-1]))
#     image_11 = Image.frombytes(image_1.mode, image_1.size, image_1.data,"raw",image_1.mode, image_1.stride)
#     image_11.save(os.path.join(path_all_photo,subject_name[j]+"_left_ear.png"))
#     ima=cv2.imread(os.path.join(path_all_photo,subject_name[j]+"_left_ear.png"))
#     ima=cv2.rotate(ima,cv2.ROTATE_90_COUNTERCLOCKWISE)
#     cv2.imwrite(os.path.join(path_all_photo,subject_name[j]+"_left_ear.png"),ima)
#     image_2= pyheif.read(os.path.join(path_raw_photo, subject_name[j],"PHOTOGRAMMETRY", list_image[35]))
#     image_22 = Image.frombytes(image_2.mode, image_2.size, image_2.data,"raw",image_2.mode, image_2.stride)
#     image_22.save(os.path.join(path_all_photo,subject_name[j]+"_right_ear.png"))
#     ima2=cv2.imread(os.path.join(path_all_photo,subject_name[j]+"_right_ear.png"))
#     ima2=cv2.rotate(ima2,cv2.ROTATE_90_COUNTERCLOCKWISE)
#     cv2.imwrite(os.path.join(path_all_photo,subject_name[j]+"_right_ear.png"),ima2)
# classifier_right =cv2.CascadeClassifier('/home/PhilipponA/Script_model_HRTF_Sonicom/data_preprocessing/haarcascade_mcs_rightear.xml')
# classifier_left =cv2.CascadeClassifier('/home/PhilipponA/Script_model_HRTF_Sonicom/data_preprocessing/haarcascade_mcs_leftear.xml')
# N=40
# for k in tqdm.trange(len(subject_name),desc="Convert PNG to ear photo"):
#     ima=cv2.imread(os.path.join(path_all_photo,subject_name[k]+"_right_ear.png"))
#     ima2=cv2.imread(os.path.join(path_all_photo,subject_name[k]+"_left_ear.png"))
#     boxes_right = classifier_left.detectMultiScale(ima)
#     boxes_left = classifier_right.detectMultiScale(ima2)
#     if len(boxes_right)!=0 :
#         if boxes_right[0][2]>=100 or boxes_right[0][3]>=100:
#             ima_ear_right=ima[boxes_right[0][1]-N:boxes_right[0][1]+boxes_right[0][3]+N,boxes_right[0][0]-N:boxes_right[0][0]+boxes_right[0][2]+N,:]
#             cv2.imwrite(os.path.join(path_ear_photo,subject_name[k]+"_right_ear.png"),ima_ear_right)
#         else:
#             list_image=os.listdir(os.path.join(path_raw_photo, subject_name[k],"PHOTOGRAMMETRY"))
#             list_image.sort()
#             image_1= pyheif.read(os.path.join(path_raw_photo, subject_name[k],"PHOTOGRAMMETRY", list_image[-1]))
#             image_11 = Image.frombytes(image_1.mode, image_1.size, image_1.data,"raw",image_1.mode, image_1.stride)
#             image_11.save(os.path.join(path_all_photo,subject_name[k]+"_left_ear.png"))
#             ima=cv2.imread(os.path.join(path_all_photo,subject_name[k]+"_left_ear.png"))
#             ima=cv2.rotate(ima,cv2.ROTATE_90_COUNTERCLOCKWISE)
#             cv2.imwrite(os.path.join(path_all_photo,subject_name[k]+"_left_ear.png"),ima)
#             ima=cv2.imread(os.path.join(path_all_photo,subject_name[k]+"_left_ear.png"))
#             boxes_right = classifier_left.detectMultiScale(ima)
#     elif len(boxes_right)==0:
#         list_image=os.listdir(os.path.join(path_raw_photo, subject_name[k],"PHOTOGRAMMETRY"))
#         list_image.sort()
#         image_1= pyheif.read(os.path.join(path_raw_photo, subject_name[k],"PHOTOGRAMMETRY", list_image[0]))
#         image_11 = Image.frombytes(image_1.mode, image_1.size, image_1.data,"raw",image_1.mode, image_1.stride)
#         image_11.save(os.path.join(path_all_photo,subject_name[k]+"_left_ear.png"))
#         ima=cv2.imread(os.path.join(path_all_photo,subject_name[k]+"_left_ear.png"))
#         ima=cv2.rotate(ima,cv2.ROTATE_90_COUNTERCLOCKWISE)
#         cv2.imwrite(os.path.join(path_all_photo,subject_name[k]+"_left_ear.png"),ima)
#         ima=cv2.imread(os.path.join(path_all_photo,subject_name[k]+"_left_ear.png"))
#         boxes_right = classifier_left.detectMultiScale(ima)
#         if len(boxes_right)!=0:
#             if boxes_right[0][2]>=100 or boxes_right[0][3]>=100:
#                 ima_ear_right=ima[boxes_right[0][1]-N:boxes_right[0][1]+boxes_right[0][3]+N,boxes_right[0][0]-N:boxes_right[0][0]+boxes_right[0][2]+N,:]
#                 cv2.imwrite(os.path.join(path_ear_photo,subject_name[k]+"_right_ear.png"),ima_ear_right)
#     if len(boxes_left)!=0:
#         if boxes_left[0][2]>=100 or boxes_left[0][3]>=100:
#             ima_ear_left=ima2[boxes_left[0][1]-N:boxes_left[0][1]+boxes_left[0][3]+N,boxes_left[0][0]-N:boxes_left[0][0]+boxes_left[0][2]+N,:]
#             cv2.imwrite(os.path.join(path_ear_photo,subject_name[k]+"_left_ear.png"),ima_ear_left)
#         else:
#             list_image=os.listdir(os.path.join(path_raw_photo, subject_name[k],"PHOTOGRAMMETRY"))
#             list_image.sort()
#             image_2= pyheif.read(os.path.join(path_raw_photo, subject_name[k],"PHOTOGRAMMETRY", list_image[35]))
#             image_22 = Image.frombytes(image_2.mode, image_2.size, image_2.data,"raw",image_2.mode, image_2.stride)
#             image_22.save(os.path.join(path_all_photo,subject_name[k]+"_right_ear.png"))
#             ima2=cv2.imread(os.path.join(path_all_photo,subject_name[k]+"_right_ear.png"))
#             ima2=cv2.rotate(ima2,cv2.ROTATE_90_COUNTERCLOCKWISE)
#             cv2.imwrite(os.path.join(path_all_photo,subject_name[k]+"_right_ear.png"),ima2)
#             ima2=cv2.imread(os.path.join(path_all_photo,subject_name[k]+"_right_ear.png"))
#             boxes_left = classifier_right.detectMultiScale(ima2)
#     elif len(boxes_left)==0 :
#         list_image=os.listdir(os.path.join(path_raw_photo, subject_name[k],"PHOTOGRAMMETRY"))
#         list_image.sort()
#         image_2= pyheif.read(os.path.join(path_raw_photo, subject_name[k],"PHOTOGRAMMETRY", list_image[35]))
#         image_22 = Image.frombytes(image_2.mode, image_2.size, image_2.data,"raw",image_2.mode, image_2.stride)
#         image_22.save(os.path.join(path_all_photo,subject_name[k]+"_right_ear.png"))
#         ima2=cv2.imread(os.path.join(path_all_photo,subject_name[k]+"_right_ear.png"))
#         ima2=cv2.rotate(ima2,cv2.ROTATE_90_COUNTERCLOCKWISE)
#         cv2.imwrite(os.path.join(path_all_photo,subject_name[k]+"_right_ear.png"),ima2)
#         ima2=cv2.imread(os.path.join(path_all_photo,subject_name[k]+"_right_ear.png"))
#         boxes_left = classifier_right.detectMultiScale(ima2)
#         if len(boxes_left)!=0:
#             if boxes_left[0][2]>=100 or boxes_left[0][3]>=100:
#                 ima_ear_left=ima2[boxes_left[0][1]-N:boxes_left[0][1]+boxes_left[0][3]+N,boxes_left[0][0]-N:boxes_left[0][0]+boxes_left[0][2]+N,:]
#                 cv2.imwrite(os.path.join(path_ear_photo,subject_name[k]+"_left_ear.png"),ima_ear_left)

for j in tqdm.trange(len(subject_name),desc="resize image to 224x224"):
    ima=cv2.imread(os.path.join(path_ear_photo,subject_name[j]+"_right_ear.png"))
    ima2=cv2.imread(os.path.join(path_ear_photo,subject_name[j]+"_left_ear.png"))
    ima=cv2.resize(ima,(224,224),interpolation=cv2.INTER_CUBIC)
    ima2=cv2.resize(ima2,(224,224), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(path_ear_resized_photo,subject_name[j]+"_right_ear.png"),ima)
    cv2.imwrite(os.path.join(path_ear_resized_photo,subject_name[j]+"_left_ear.png"),ima2)