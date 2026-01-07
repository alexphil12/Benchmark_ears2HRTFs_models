"""
This script processes 2D ear images to extract landmarks and compute ear characteristic distances 
using a pre-trained TensorFlow model. The results are saved as images with landmarks overlaid 
and as a NumPy array containing the computed distances.
Modules:
        - tensorflow: For loading and using the pre-trained model.
        - os: For file and directory operations.
        - matplotlib.pyplot: For image visualization and saving.
        - matplotlib.image: For reading image files.
        - numpy: For numerical operations.
        - load_data_landmark_model: Custom module for data loading (commented out in this script).
        - tensorboard: For TensorBoard integration (not used in this script).
        - tqdm: For progress bar visualization.
        - copy: For deep copying objects.
        - keras.applications.imagenet_utils: For preprocessing input images.
Functions:
        - None explicitly defined. The script runs as a standalone process.
Workflow:
        1. Load a pre-trained TensorFlow model for ear landmark detection.
        2. Iterate through a list of subjects and process their ear images:
                - Load and preprocess the images.
                - Use the model to predict landmarks for both left and right ears.
                - Compute ear characteristic distances for the left ear:
                        0: Cavum concha height
                        1: Cymba concha height
                        2: Cavum concha width
                        3: Fossa height
                        4: Pinna height
                        5: Pinna width
                        6: Intertragal incisure width
                - Overlay landmarks on the original images and save the results.
        3. Save the computed distances as a NumPy array.
Inputs:
        - 2D ear images stored in the specified directories.
        - Pre-trained TensorFlow model for landmark detection.
Outputs:
        - Images with landmarks overlaid, saved in the specified directory.
        - NumPy array containing computed ear characteristic distances.
Notes:
        - Ensure the paths to the images and model are correctly set before running the script.
        - The script assumes specific naming conventions for the input images.
        - The model file should be compatible with TensorFlow's `load_model` function.
"""
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
from load_data_landmark_model import dataload
import tensorboard as tb
from tqdm import trange
import copy as cp
from keras.applications.imagenet_utils import preprocess_input

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
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

Ear_marking_model = tf.keras.models.load_model('/home/PhilipponA/Imported-models/my_model.h5')
Ear_marking_model.summary()
# X_test, y_test = dataload(test=True, size=630, test_size=630)
# loss, acc = Ear_marking_model.evaluate(X_test, y_test, verbose=2)
# print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
ear_distances=np.zeros((len(subject_name),7))
ear_distances_right=np.zeros((len(subject_name),7))
image_dimensions_left=np.zeros((len(subject_name),2)) # to store the image dimensions for each subject
image_dimensions_right=np.zeros((len(subject_name),2)) # to store the image dimensions for each subject
for N_s in trange(len(subject_name),desc="Processing images to get landmarks and distances"):
        ima_right=plt.imread(os.path.join(path_ear_resized_photo,subject_name[N_s]+"_right_ear.png"))
        ima_left=plt.imread(os.path.join(path_ear_resized_photo,subject_name[N_s]+"_left_ear.png"))
        ima_right=ima_right[:,:,0:3]
        ima_left=ima_left[:,:,0:3]
        ima_right=ima_right[None,:]* 255.0
        ima_left=ima_left[None,:]* 255.0
        ima_right=preprocess_input(ima_right)
        ima_left=preprocess_input(ima_left)
        ima_right=np.take(ima_right,[0,1,2],axis=3)
        ima_left=np.take(ima_left,[0,1,2],axis=3)
        landmark_right=np.squeeze(Ear_marking_model(ima_right,training=False))
        landmark_right=cp.copy((landmark_right * 224).astype(np.int32))
        landmark_left=np.squeeze(Ear_marking_model(ima_left,training=False))
        landmark_left=cp.copy((landmark_left * 224).astype(np.int32))
        
        ori_ear_photo=img.imread(os.path.join(path_ear_photo,subject_name[N_s]+"_left_ear.png"))
        shape_ear_photo=ori_ear_photo.shape
        image_dimensions_left[N_s,0]=shape_ear_photo[0] # height
        image_dimensions_left[N_s,1]=shape_ear_photo[1] # width
        new_landmark_left=cp.copy(landmark_left)
        new_landmark_left[0:55]=new_landmark_left[0:55]*shape_ear_photo[0]
        new_landmark_left[55:110]=new_landmark_left[55:110]*shape_ear_photo[1]
        """ Computing of the ears characteristics distances, they correspond form 0 to 6:
         0: cavum concha height
         1: cymba concha height
         2: cavum concha width
         3: fossa height
         4: pinna height
         5: pinna width
         6: intertragal incisure width"""
        ear_distances[N_s,0]=np.sqrt((new_landmark_left[19]-new_landmark_left[38])**2+(new_landmark_left[19+55]-new_landmark_left[38+55])**2)
        ear_distances[N_s,1]=np.sqrt((new_landmark_left[19]-new_landmark_left[47])**2+(new_landmark_left[19+55]-new_landmark_left[47+55])**2)
        ear_distances[N_s,2]=np.sqrt((new_landmark_left[36]-new_landmark_left[42])**2+(new_landmark_left[36+55]-new_landmark_left[42+55])**2)
        ear_distances[N_s,3]=np.sqrt((new_landmark_left[24]-new_landmark_left[47])**2+(new_landmark_left[24+55]-new_landmark_left[47+55])**2)
        ear_distances[N_s,4]=np.sqrt((new_landmark_left[3]-new_landmark_left[17])**2+(new_landmark_left[3+55]-new_landmark_left[17+55])**2)
        ear_distances[N_s,5]=np.sqrt((new_landmark_left[32]-new_landmark_left[36])**2+(new_landmark_left[32+55]-new_landmark_left[36+55])**2)
        ear_distances[N_s,6]=np.sqrt((new_landmark_left[37]-new_landmark_left[39])**2+(new_landmark_left[37+55]-new_landmark_left[39+55])**2)

        ori_ear_photo_right=img.imread(os.path.join(path_ear_photo,subject_name[N_s]+"_right_ear.png"))
        shape_ear_photo_r=ori_ear_photo_right.shape
        image_dimensions_right[N_s,0]=shape_ear_photo_r[0] # height
        image_dimensions_right[N_s,1]=shape_ear_photo_r[1] # width
        new_landmark_right=cp.copy(landmark_right)
        new_landmark_right[0:55]=new_landmark_right[0:55]*shape_ear_photo_r[0]
        new_landmark_right[55:110]=new_landmark_right[55:110]*shape_ear_photo_r[1]
        fit_coef=(np.corrcoef(new_landmark_right[0:55],new_landmark_right[55:110])**2)[0,1]
        # print("fit_coef: {coef} ".format(coef=fit_coef))
        if fit_coef>0.25:
               ear_distances_right[N_s,:]=ear_distances[N_s,:] #Some right ear are no well landmarked. It should correct it for the most devious cases
        else:
               ear_distances_right[N_s,0]=np.sqrt((new_landmark_right[19]-new_landmark_right[38])**2+(new_landmark_right[19+55]-new_landmark_right[38+55])**2)
               ear_distances_right[N_s,1]=np.sqrt((new_landmark_right[19]-new_landmark_right[47])**2+(new_landmark_right[19+55]-new_landmark_right[47+55])**2)
               ear_distances_right[N_s,2]=np.sqrt((new_landmark_right[36]-new_landmark_right[42])**2+(new_landmark_right[36+55]-new_landmark_right[42+55])**2)
               ear_distances_right[N_s,3]=np.sqrt((new_landmark_right[24]-new_landmark_right[47])**2+(new_landmark_right[24+55]-new_landmark_right[47+55])**2)
               ear_distances_right[N_s,4]=np.sqrt((new_landmark_right[3]-new_landmark_right[17])**2+(new_landmark_right[3+55]-new_landmark_right[17+55])**2)
               ear_distances_right[N_s,5]=np.sqrt((new_landmark_right[32]-new_landmark_right[36])**2+(new_landmark_right[32+55]-new_landmark_right[36+55])**2)
               ear_distances_right[N_s,6]=np.sqrt((new_landmark_right[37]-new_landmark_right[39])**2+(new_landmark_right[37+55]-new_landmark_right[39+55])**2)
        
        img_original_right = plt.imread(os.path.join(path_ear_resized_photo,subject_name[N_s]+"_right_ear.png"))
        for j in range(0,55):  # drop the landmark points on the image
                plt.scatter([landmark_right[j]], [landmark_right[j+55]])
        plt.imshow(img_original_right)
        plt.savefig(os.path.join(path_ear_resized_photo,subject_name[N_s]+"_right_ear_landmark.png"))
        plt.close()

        img_original_left = plt.imread(os.path.join(path_ear_resized_photo,subject_name[N_s]+"_left_ear.png"))
        for j in range(0,55):  # drop the landmark points on the image
                plt.scatter([landmark_left[j]], [landmark_left[j+55]])
        plt.imshow(img_original_left)
        plt.savefig(os.path.join(path_ear_resized_photo,subject_name[N_s]+"_left_ear_landmark.png"))
        plt.close()
        np.save('/home/PhilipponA/HRTF_sonicom_dataset/Measure_1D_ears/Ear_distances_1D/Ear_distances_left_sonicom_' + subject_name[N_s] , ear_distances[N_s])
        np.save('/home/PhilipponA/HRTF_sonicom_dataset/Measure_1D_ears/Ear_distances_1D/Ear_distances_right_sonicom_'+ subject_name[N_s], ear_distances_right[N_s])

np.save('/home/PhilipponA/HRTF_sonicom_dataset/Measure_1D_ears/Ear_distances_1D/Ear_distances_left_sonicom', ear_distances)
np.save('/home/PhilipponA/HRTF_sonicom_dataset/Measure_1D_ears/Ear_distances_1D/Ear_distances_right_sonicom', ear_distances_right)


np.save('/home/PhilipponA/HRTF_sonicom_dataset/Measure_1D_ears/Ear_distances_1D/image_dimensions_left_sonicom.npy', image_dimensions_left)
np.save('/home/PhilipponA/HRTF_sonicom_dataset/Measure_1D_ears/Ear_distances_1D/image_dimensions_right_sonicom.npy', image_dimensions_right)