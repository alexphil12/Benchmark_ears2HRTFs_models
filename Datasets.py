# Datasets
import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Lambda
from pysofaconventions import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from utils import Calculate_SHT_coef


def cartesian_to_spherical(x,y,z):
    r = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arccos(z / r)  # polar angle
    theta = np.arctan2(y, x)   # azimuthal angle
    return r, theta, phi

def read_hrtf(self, HRTF_path, output_type="IR_dir", Y_basis=None, device="cpu", out_HRTF=False):
    if not os.path.exists(HRTF_path):
        raise FileNotFoundError(f"HRTF file not found: {HRTF_path}")

    HRTF_y = None
    try:
        # Ouvre le fichier (chaque appel ouvre+lit+ferme)
        HRTF_y = SOFAFile(HRTF_path, 'r')

        # Lecture et COPIE immédiate pour détacher de l'objet C sous-jacent
        HRTF_y_IR = HRTF_y.getDataIR()
        # getDataIR peut retourner un numpy array view sur une zone C : on force la copie
        HRTF_y_IR = np.array(HRTF_y_IR, copy=True)

        if output_type in ("IR_dir", "IR_set"):
            out = np.array(HRTF_y_IR, dtype=np.float32, copy=True)

        elif output_type in ("HRTF_dir", "HRTF_set", "HRTF_duty"):
            HRTF_y_set = 20 * np.log10(np.abs(np.fft.fft(HRTF_y_IR, axis=2)) + 1e-4)
            HRTF_y_set = HRTF_y_set[0:793, :, 0:HRTF_y_set.shape[2] // 2]
            out = np.array(HRTF_y_set, dtype=np.float32, copy=True)

        elif output_type in ("SHT_set", "SHT_freq"):
            HRTF_y_set = 20 * np.log10(np.abs(np.fft.fft(HRTF_y_IR, axis=2)) + 1e-4)
            HRTF_y_set_L = HRTF_y_set[0:793, 0, 0:HRTF_y_set.shape[2] // 2]
            HRTF_y_set_R = HRTF_y_set[0:793, 1, 0:HRTF_y_set.shape[2] // 2]

            # Convert to torch on the requested device (no numpy view left)
            HRTF_y_set_L = torch.tensor(HRTF_y_set_L, dtype=torch.complex64, device=device)
            HRTF_y_set_R = torch.tensor(HRTF_y_set_R, dtype=torch.complex64, device=device)

            from utils import Calculate_SHT_coef
            SHT_coef = Calculate_SHT_coef(HRTF_y_set_L, HRTF_y_set_R, Y_basis, alpha=0, device=device)
            out = SHT_coef  # torch tensor (complex64)

        else:
            raise ValueError(f"Unknown output_type: {output_type}")

        if out_HRTF:
            HRTF_y_set = 20 * np.log10(np.abs(np.fft.fft(HRTF_y_IR, axis=2)) + 1e-4)
            HRTF_y_set = HRTF_y_set[0:793, :, 0:HRTF_y_set.shape[2] // 2]
            out = np.array(HRTF_y_set, dtype=np.float32, copy=True)

        return out

    finally:
        # fermer explicitement pour éviter que GC (dans un autre thread) manipule l'objet en mauvais état
        if HRTF_y is not None:
            try:
                HRTF_y.close()
            except Exception:
                # ignore close errors but don't laisser l'exception originale se perdre
                pass



To_Tensor = Lambda(lambda y: torch.from_numpy(y))


class HRTF_mesh_dataset(Dataset):
    def __init__(self, ears_dataset_path="/databases/HRTF_sonicom_dataset/",HRTF_dataset_path="/data/HRTF_dataset", type_of_data="1d" ,output_type="IR_dir", transform=To_Tensor, target_transform=To_Tensor,Train_data=True,Test_data=True,L=8,mode="left",out_HRTF=False,distorted_photo=False,percent_of_data=80,Y_basis="oups",device="cpu"):
        self.device=device
        self.Y_basis=Y_basis       
        self.ears_dataset_path = ears_dataset_path
        self.HRTF_dataset_path=HRTF_dataset_path
        self.train_mode=Train_data
        self.distorted_photo=distorted_photo
        self.subject_list=os.listdir(ears_dataset_path+"/Mesh_3D_ears/STL_mesh/")
        self.subject_list=sorted(self.subject_list)
        self.subject_list=self.subject_list[0:len(self.subject_list):2]
        self.subject_list=[x[0:5] for x in self.subject_list]
        if self.train_mode:
            self.subject_list=self.subject_list[0:((len(self.subject_list)*percent_of_data)//100)]
        else:
            if Test_data:
                self.subject_list=self.subject_list[((len(self.subject_list)*percent_of_data)//100):(len(self.subject_list)*(percent_of_data +10))//100] 
            else:
                self.subject_list=self.subject_list[((len(self.subject_list)*(percent_of_data +10))//100):len(self.subject_list)]                                                                                                                                                 
        self.transform = transform
        self.target_transform = target_transform
        self.data_type=type_of_data
        self.data_1D=np.load(self.ears_dataset_path+"/Measure_1D_ears/Ear_distances_1D/Ear_distances_left_sonicom.npy")
        self.max_1D=np.max(self.data_1D,axis=0)
        self.min_1D=np.min(self.data_1D,axis=0)
        self.output_type=output_type
        if out_HRTF:
            self.output_type="HRTF_duty"
        HRTF_y=SOFAFile(os.path.join(self.HRTF_dataset_path,self.subject_list[0],"HRTF","HRTF","44kHz",self.subject_list[0]+"_Windowed_44kHz.sofa"),'r')
        self.incidence_dir=HRTF_y.getVariableValue('SourcePosition')
        self.incidence_dir=self.incidence_dir[:,0:2]
        self.incidence_dir=self.incidence_dir.astype(np.float32)
        self.incidence_dir=torch.from_numpy(self.incidence_dir)
        self.order_of_sht=L
        self.mode=mode

    def __len__(self):
        return len(self.subject_list)

    def __getitem__(self, idx,out_HRTF=False):
        HRTF_path = os.path.join(self.HRTF_dataset_path,self.subject_list[idx],"HRTF","HRTF","44kHz",self.subject_list[idx]+"_Windowed_44kHz.sofa")
        HRTF_y_IR = read_hrtf(self,HRTF_path,output_type=self.output_type,out_HRTF=out_HRTF,Y_basis=self.Y_basis,device=self.device)
        if self.data_type=="1d":
            X_data_R=np.load(self.ears_dataset_path+"/Measure_1D_ears/Ear_distances_1D/Ear_distances_right_sonicom_"+self.subject_list[idx]+".npy")
            X_data_L=np.load(self.ears_dataset_path+"/Measure_1D_ears/Ear_distances_1D/Ear_distances_left_sonicom_"+self.subject_list[idx]+".npy")
            X_data=np.stack((X_data_L,X_data_R),axis=0)
            X_data=np.expand_dims(X_data, axis=2)
            X_data_L = (X_data_L - self.min_1D) / (self.max_1D - self.min_1D)
            X_data_R = (X_data_R - self.min_1D) / (self.max_1D - self.min_1D)
            X_data=np.stack((X_data_L,X_data_R),axis=0,dtype=np.float32)
            X_data_L=np.expand_dims(X_data_L, axis=0)
            X_data_R=np.expand_dims(X_data_R, axis=0)
            if self.mode=="left":
                X_data_end=X_data_L.astype(np.float32)
            elif self.mode=="right":
                X_data_end=X_data_R.astype(np.float32)
            elif self.mode=="both":
                X_data_end=X_data
            else:
                raise ValueError("Invalid mode. Choose 'left', 'right', or 'both'.")

        elif self.data_type=="2d":
            if self.distorted_photo:
                X_data_R=plt.imread(self.ears_dataset_path+"/Photo_2D_ears/PNG_ear_resized_photo/"+self.subject_list[idx]+"_right_ear.png")
                X_data_L=plt.imread(self.ears_dataset_path+"/Photo_2D_ears/PNG_ear_resized_photo/"+self.subject_list[idx]+"_left_ear.png")
            else:
                X_data_R=plt.imread(self.ears_dataset_path+"/Photo_2D_ears/PNG_ear_photo_no_disto/"+self.subject_list[idx]+"_right_ear.png")
                X_data_L=plt.imread(self.ears_dataset_path+"/Photo_2D_ears/PNG_ear_photo_no_disto/"+self.subject_list[idx]+"_left_ear.png")
            X_data_L=np.moveaxis(X_data_L, 2, 0)
            X_data_R=np.moveaxis(X_data_R, 2, 0)
            X_data=np.concatenate((X_data_L,X_data_R),axis=0)
            X_data_L=X_data_L.astype(np.float32)
            X_data_R=X_data_R.astype(np.float32)
            X_data=X_data.astype(np.float32)
            if self.mode=="left":
                X_data_end=X_data_L
            elif self.mode=="right":
                X_data_end=X_data_R
            elif self.mode=="both":
                X_data_end=X_data
            else:
                raise ValueError("Invalid mode. Choose 'simple_left', 'simple_right', or 'Both'.")

        elif self.data_type=="3d":
            X_data_R=np.load(self.ears_dataset_path+"/Mesh_3D_ears/Grid_sampled_data/"+self.subject_list[idx]+"_right_ear_grid.npy")
            X_data_L=np.load(self.ears_dataset_path+"/Mesh_3D_ears/Grid_sampled_data/"+self.subject_list[idx]+"_left_ear_grid.npy")
            X_data_L=X_data_L.astype(np.float32)
            X_data_R=X_data_R.astype(np.float32)
            X_data_L=np.expand_dims(X_data_L, axis=0)
            X_data_R=np.expand_dims(X_data_R, axis=0)
            X_data=np.concatenate((X_data_L,X_data_R),axis=0)
            if self.mode=="left":
                X_data_end=X_data_L
            elif self.mode=="right":
                X_data_end=X_data_R
            elif self.mode=="both":
                X_data_end=X_data
            else:
                raise ValueError("Invalid mode. Choose 'simple_left', 'simple_right', or 'Both'.")
        else:
            raise ValueError("Invalid data type. Choose '1D', '2D', or '3D'.")
        if self.target_transform:
            HRTF_y_IR = self.target_transform(HRTF_y_IR)
        if self.transform:
            X_data_end = self.transform(X_data_end)
        return  X_data_end,HRTF_y_IR
    
if __name__ == "__main__":
    HRTF_dataset=HRTF_mesh_dataset(type_of_data="2d",output_type="HRTF_dir",Train_data=True,L=17,mode="left",out_HRTF=False,distorted_photo=False)
    incidence=HRTF_dataset.incidence_dir
    x_data,HRTF = HRTF_dataset.__getitem__(0)
    torch.save(incidence,"incidence.pt")
    uwu=0
    