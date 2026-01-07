import torch
import torch.nn as nn
import torch.nn.functional as F

class Woo_lee_Model(nn.Module):
    def __init__(self,input_type='1d',ear_input='left',dropout_rates=[0.1, 0.1, 0.1],distorted_photo=False):
        self.ear_input=ear_input
        self.distorted_photo=distorted_photo
        if self.ear_input == 'both':
            N_channels=2
        else:
            N_channels=1
        self.input_type = input_type
        super(Woo_lee_Model, self).__init__()
        self.dropout_1 = nn.Dropout(dropout_rates[0])
        self.dropout_2 = nn.Dropout(dropout_rates[1])
        self.dropout_3 = nn.Dropout(dropout_rates[2])

        if self.input_type == '1d':
            self.dir_input= nn.Linear(2,500)
            self.input = nn.Conv1d(N_channels,16,kernel_size=3,stride=1,padding=1)
            self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
            self.cnn = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
            self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)
            self.flat= nn.Flatten()
            self.fc2=nn.Linear(32,500)
            self.batch_norm = nn.BatchNorm1d(128)
            self.lin_end1=nn.Linear(1000,128)
            self.lin_end=nn.Linear(128,128)
            self.end=nn.Linear(128,512)
        elif self.input_type == '2d':
            self.dir_input= nn.Linear(2,500)

            self.input = nn.Conv2d(3*N_channels, 8, kernel_size=3, stride=1, padding=1)
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.cnn = nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1)
            self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

            self.input_b = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.maxpool_b = nn.MaxPool2d(kernel_size=2, stride=2)
            self.cnn_b = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
            self.maxpool2_b = nn.MaxPool2d(kernel_size=2, stride=2)

            self.flat= nn.Flatten()
            self.batch_norm = nn.BatchNorm1d(500)
            if self.distorted_photo== False:
                self.fc2 =nn.Sequential(nn.Linear(113664, 1000), nn.LeakyReLU(negative_slope=0.1), nn.Linear(1000, 500))
            else:
                self.fc2 = nn.Linear(25088, 500)
            self.lin_end1=nn.Linear(1000,500)
            self.lin_end=nn.Linear(500,500)
            self.end=nn.Linear(500,512)

        elif self.input_type == '3d':
            self.dir_input= nn.Linear(2,500)

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
            self.batch_norm = nn.BatchNorm1d(500)
            self.fc2 = nn.Linear(1024, 500)
            self.lin_end1=nn.Linear(1000,500)
            self.lin_end=nn.Linear(500,500)
            self.end=nn.Linear(500,512)
        else:
            raise ValueError("Invalid input type. Choose '1d', '2d', or '3d'.")

    def forward(self, x,direction,out_HRTF=False):
        device= x.device
        if self.input_type == '1d':
            x = self.input(x)
            x = self.maxpool(x)
            x = F.leaky_relu(x,0.1)
            x = self.dropout_1(x)
            x = self.cnn(x)
            x = self.maxpool2(x)
            x = F.leaky_relu(x,0.1)
            x = self.dropout_2(x)
        elif self.input_type == '2d':
            x = self.input(x)
            x = self.maxpool(x)
            x = F.leaky_relu(x,0.1)
            x = self.dropout_1(x)
            x = self.cnn(x)
            x = self.maxpool2(x)
            x = F.leaky_relu(x,0.1)
            x = self.dropout_2(x)
            x = self.input_b(x)
            x = self.maxpool_b(x)
            x = F.leaky_relu(x,0.1)
            x = self.cnn_b(x)
            x = self.maxpool2_b(x)
            x = F.leaky_relu(x,0.1)
        elif self.input_type == '3d':
            x = self.input(x)
            x = F.leaky_relu(x,0.1)
            x = self.dropout_1(x)
            x = self.cnn(x)
            x = self.maxpool2(x)
            x = F.leaky_relu(x,0.1)
            x = self.dropout_2(x)
            x = self.input_b(x)
            x = self.maxpool_b(x)
            x = F.leaky_relu(x,0.1)
            x = self.cnn_b(x)
            x = F.leaky_relu(x,0.1)
            x = self.maxpool2_b(x)
            x = self.input_c(x)
            x = self.maxpool_c(x)
            x = F.leaky_relu(x,0.1)
            x = self.cnn_c(x)
            x = self.maxpool2_c(x)
            x = F.leaky_relu(x,0.1)
        dir_embeded = self.dir_input(direction)
        dir_embeded = F.leaky_relu( dir_embeded,0.1)
        x = self.flat(x)
        x = self.fc2(x)
        x = F.leaky_relu(x,0.1)
        X_dir = torch.cat((x,dir_embeded),1) 
        x = self.lin_end1(X_dir)
        x = F.leaky_relu(x,0.1)
        x = self.lin_end(x)
        x = F.leaky_relu(x,0.1)
        x = self.dropout_3(x)
        x = self.lin_end(x)
        x = F.leaky_relu(x,0.1)
        x = self.lin_end(x)
        x = F.leaky_relu(x,0.1)
        x = self.end(x)
        x= F.relu(x)
        if out_HRTF==True:
            left_mag = 20*torch.log10(torch.abs(torch.fft.fft(x[:,0:256]))+1e-10)
            right_mag = 20*torch.log10(torch.abs(torch.fft.fft(x[:,256:512]))+1e-10)
            left_mag = left_mag[:,0:128]
            right_mag = right_mag[:,0:128]
            x = torch.cat((left_mag,right_mag),1)

        return x

# Example usage
if __name__ == "__main__":
    model = Woo_lee_Model( input_type='1d')

    # Dummy input
    input_tensor = torch.randn(20,1,7)
    direction_input=torch.randn(20,2)
    output = model(input_tensor,direction_input)
    print(output.shape)