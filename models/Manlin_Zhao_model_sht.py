import torch
import torch.nn as nn
import torch.nn.functional as F
from pysofaconventions import *
import os
import numpy as np

class Manlin_Zhao_Model(nn.Module):
    def __init__(self,input_type='1d',order_of_sht=4,num_of_frequency=128,ear_input='left',incidence="oups",distorted_photo=False,Y_basis="oups"):
        super(Manlin_Zhao_Model, self).__init__() 
        self.input_type = input_type
        self.output_sizes = (1,2,int((order_of_sht+1)**2),2) # num_freq, left/right ear,(lmax+1)²,real/imag 
        self.ear_input=ear_input
        self.order_of_sht=order_of_sht
        self.Y_basis=Y_basis
        self.incidence = incidence
        self.freq_embed=nn.Sequential(nn.Linear(128,500),nn.ReLU())
        if self.ear_input == 'both':
            N_channels=2
        else:
            N_channels=1
        if self.input_type == '1d':
            self.input = nn.Conv1d(N_channels,16,kernel_size=3,stride=1,padding=1)
            self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
            self.cnn = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
            self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)
            self.fc2_left=nn.Linear(32,1000)
            self.fc2_right=nn.Linear(32,1000)
            self.lin_end1_left=nn.Linear(1000,500)
            self.lin_end1_right=nn.Linear(1000,500)
            self.lin_end_left=nn.Linear(1000,32)
            self.lin_end_right=nn.Linear(1000,32)
            self.batch_norm = nn.BatchNorm1d(1000)
            self.batch_norm2= nn.BatchNorm1d(32)
            self.batch_norm3= nn.BatchNorm1d(32)
        elif self.input_type == '2d':
            self.input = nn.Conv2d(3*N_channels, 8, kernel_size=3, stride=2, padding=1)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
            self.cnn = nn.Conv2d(8, 32, kernel_size=3, stride=2, padding=1)
            self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=1)
            self.input_b = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.maxpool_b = nn.MaxPool2d(kernel_size=3, stride=2)
            self.cnn_b = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
            self.maxpool2_b = nn.MaxPool2d(kernel_size=3, stride=2)

            self.flat= nn.Flatten()
            if distorted_photo == False:
                self.fc2_left =nn.Sequential(nn.Linear(23936, 10000), nn.LeakyReLU(negative_slope=0.1), nn.Linear(10000, 1000))
                self.fc2_right =nn.Sequential(nn.Linear(23936, 10000), nn.LeakyReLU(negative_slope=0.1), nn.Linear(10000, 1000))
            else:
                self.fc2_left = nn.Linear(3200, 1000)
                self.fc2_right = nn.Linear(3200, 1000)
            self.lin_end1_left=nn.Linear(1000,500)
            self.lin_end_left=nn.Linear(1000,32)
            self.lin_end1_right=nn.Linear(1000,500)
            self.lin_end_right=nn.Linear(1000,32)
            self.batch_norm = nn.BatchNorm1d(1000)
            self.batch_norm2= nn.BatchNorm2d(32)
            self.batch_norm3= nn.BatchNorm1d(32)

        elif self.input_type == '3d':
            self.input = nn.Conv3d(N_channels, 8, kernel_size=3, stride=1, padding=1)
            self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
            self.cnn = nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=1)
            self.maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2)
            self.input_b = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1)
            self.maxpool_b = nn.MaxPool3d(kernel_size=2, stride=2)
            self.cnn_b = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
            self.maxpool2_b = nn.MaxPool3d(kernel_size=2, stride=2)
            self.input_c = nn.Conv3d(64, 96, kernel_size=3, stride=1, padding=1)
            self.maxpool_c = nn.MaxPool3d(kernel_size=2, stride=2)
            self.cnn_c = nn.Conv3d(96, 128, kernel_size=3, stride=1, padding=1)
            self.maxpool2_c = nn.MaxPool3d(kernel_size=2, stride=2)
            self.flat= nn.Flatten()
            self.fc2_left = nn.Linear(1024, 1000)
            self.lin_end1_left=nn.Linear(1000,500)
            self.lin_end_left=nn.Linear(1000,32)

            self.fc2_right = nn.Linear(1024, 1000)
            self.lin_end1_right=nn.Linear(1000,500)
            self.lin_end_right=nn.Linear(1000,32)

            self.batch_norm3= nn.BatchNorm1d(32)
        else:
            raise ValueError("Invalid input type. Choose '1d', '2d', or '3d'.")
        self.conv_end_right=nn.ConvTranspose1d(1,32, kernel_size=3, stride=1, padding=1)
        self.maxpool_end_right = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv_end2_right=nn.ConvTranspose1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_end2_right = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv_end3_right=nn.ConvTranspose1d(64, 128, kernel_size=3, stride=1, padding=1)

        self.conv_end_left=nn.ConvTranspose1d(1,32, kernel_size=3, stride=1, padding=1)
        self.maxpool_end_left = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv_end2_left=nn.ConvTranspose1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_end2_left = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv_end3_left=nn.ConvTranspose1d(64, 128, kernel_size=3, stride=1, padding=1)

        self.flat_end= nn.Flatten()
        self.end=nn.Linear(4096,self.output_sizes[0]*1*self.output_sizes[2]*self.output_sizes[3])

    def forward(self, x,freq,out_HRTF=False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        one_hot_freq = F.one_hot(freq, num_classes=128).float()
        freq_embedding = self.freq_embed(one_hot_freq)
        freq_embedding = freq_embedding.squeeze(1)

        if self.input_type == '1d':
            x = self.input(x)
            x = F.relu(x)
            x = self.maxpool(x)
            x = self.cnn(x)
            x = F.relu(x)
            x = self.maxpool2(x)
        elif self.input_type == '2d':
            x = self.input(x)
            x = F.relu(x)
            x = self.maxpool(x)
            x = self.cnn(x)
            x = F.relu(x)
            x = self.maxpool2(x)
            x = self.batch_norm2(x)
            x = self.input_b(x)
            x = F.relu(x)
            x = self.maxpool_b(x)
            x = self.cnn_b(x)
            x = F.relu(x)
            x = self.maxpool2_b(x)
        elif self.input_type == '3d':
            x = self.input(x)
            x = F.relu(x)
            x = self.maxpool(x)
            x = self.cnn(x)
            x = F.relu(x)
            x = self.input_b(x)
            x = F.relu(x)
            x = self.maxpool_b(x)
            x = self.cnn_b(x)
            x = F.relu(x)
            x = self.maxpool2_b(x)
            x = self.input_c(x)
            x = F.relu(x)
            x = self.maxpool_c(x)
            x = self.cnn_c(x)
            x = F.relu(x)
            x = self.maxpool2_c(x)

        x = self.flat_end(x)
        x1 = self.fc2_left(x)
        x1 = F.relu(x1)
        x1 = self.lin_end1_left(x1)
        x1 = F.relu(x1)
        x1 = torch.cat((x1, freq_embedding), dim=1)
        x1 = self.lin_end_left(x1)
        x1 = F.relu(x1)
        x1 = torch.reshape(x1,(x1.shape[0], 1, x1.shape[1]))

        x2 = self.fc2_right(x)
        x2 = F.relu(x2)
        x2 = self.lin_end1_right(x2)
        x2 = F.relu(x2)
        x2 = torch.cat((x2, freq_embedding), dim=1)
        x2 = self.lin_end_right(x2)
        x2 = F.relu(x2)
        x2 = torch.reshape(x2,(x2.shape[0], 1, x2.shape[1]))

        x1 = self.conv_end_right(x1)
        x1 = F.relu(x1)
        # x1 = self.maxpool_end_right(x1)
        x1 = self.conv_end2_right(x1)
        x1 = F.relu(x1)
        # x1 = self.maxpool_end2_right(x1)
        x1 = self.conv_end3_right(x1)
        x1 = F.relu(x1)
        x1 = self.flat_end(x1)
        x1 = self.end(x1)
        x1 = torch.reshape(x1,(x.shape[0],self.output_sizes[2],1,self.output_sizes[0],self.output_sizes[3]))

        x2 = self.conv_end_left(x2)
        x2 = F.relu(x2)
        # x2 = self.maxpool_end_left(x2)
        x2 = self.conv_end2_left(x2)
        x2 = F.relu(x2)
        # x2 = self.maxpool_end2_left(x2)
        x2 = self.conv_end3_left(x2)
        x2 = F.relu(x2)
        x2 = self.flat_end(x2)
        x2 = self.end(x2)
        x2 = torch.reshape(x2,(x.shape[0],self.output_sizes[2],1,self.output_sizes[0],self.output_sizes[3]))

        x = torch.cat((x1, x2), dim=2)
        x = torch.view_as_complex(x)  # Concatenate left and right ear outputs
        if out_HRTF==True:
            recon_measured = torch.zeros((x.shape[0], 793, 2), dtype=torch.float32).to(device)
            for i in range(x.shape[0]):
                for l in range(2):
                    H = self.Y_basis @ x[i, :, l, 0]
                    recon_measured[i, :, l]= H

            x = recon_measured
            


            
        return x

# Example usage
if __name__ == "__main__":
    HRTF_y=SOFAFile(os.path.join("/databases/sonicom_hrtf_dataset/","P0008","HRTF","HRTF","44kHz","P0008"+"_Windowed_44kHz.sofa"),'r')
    incidence_dir=HRTF_y.getVariableValue('SourcePosition')
    incidence_dir=incidence_dir[:,0:2]
    incidence_dir=incidence_dir.astype(np.float32)
    incidence_dir=np.array(incidence_dir)
    input_tensor = torch.randn(20,3,606,396)
    freq = torch.randint(0, 128,size=(20,1))
    model = Manlin_Zhao_Model( input_type='2d',distorted_photo=True,order_of_sht=30,num_of_frequency=128,incidence=incidence_dir,ear_input='left')
    output = model(input_tensor,freq,out_HRTF=False)
    print(output.shape)