import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MLP(nn.Module):
    def __init__(self, hidden_sizes, num_filters,input_type='1d',ear_input='left',dropout_rates=[0.001, 0.001],distorted_photo=False):
        super(MLP, self).__init__()
        self.input_type = input_type
        self.flat= nn.Flatten()
        assert(num_filters>2)
        in_embed =[]
        if ear_input == 'both':
            N_channels=2
        else:
            N_channels=1
        if input_type == '1d':
            self.input_size = 7
            in_embed.append(nn.Conv1d(N_channels,16,kernel_size=4,stride=1,padding=1))
            in_embed.append(nn.LeakyReLU())
            in_embed.append(nn.BatchNorm1d(16))
            in_embed.append(nn.Dropout(dropout_rates[0]))
            in_embed.append(nn.MaxPool1d(kernel_size=2, stride=2))
            in_embed.append(nn.Conv1d(16, 32, kernel_size=4, stride=1, padding=1))
            in_embed.append(nn.LeakyReLU())
            in_embed.append(nn.BatchNorm1d(32))
            in_embed.append(nn.Dropout(dropout_rates[1]))
            in_embed.append(nn.MaxPool1d(kernel_size=2, stride=2))
            in_embed.append(nn.Flatten())
            in_embed.append(nn.BatchNorm1d(32))
            in_embed.append(nn.Linear(32,hidden_sizes[0]))
            self.in_embed = nn.Sequential(*in_embed)
        elif input_type == '2d':
            self.input_size = 224*224
            in_embed.append(nn.Conv2d(3*N_channels,16,kernel_size=4,stride=1,padding=1))
            in_embed.append(nn.LeakyReLU())
            in_embed.append(nn.BatchNorm2d(16))
            in_embed.append(nn.Dropout(dropout_rates[0]))
            in_embed.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_embed.append(nn.Conv2d(16, 32, kernel_size=4, stride=1, padding=1))
            in_embed.append(nn.LeakyReLU())
            in_embed.append(nn.BatchNorm2d(32))
            in_embed.append(nn.Dropout(dropout_rates[1]))
            in_embed.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_embed.append(nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=1))
            in_embed.append(nn.LeakyReLU())
            in_embed.append(nn.BatchNorm2d(64))
            in_embed.append(nn.Dropout(dropout_rates[0]))
            in_embed.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_embed.append(nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=1))
            in_embed.append(nn.LeakyReLU())
            in_embed.append(nn.BatchNorm2d(128))
            in_embed.append(nn.Dropout(dropout_rates[1]))
            in_embed.append(nn.MaxPool2d(kernel_size=2, stride=2))
            if distorted_photo== False:
                in_embed.append(nn.Conv2d(128, 196, kernel_size=4, stride=1, padding=1))
                in_embed.append(nn.LeakyReLU())
                in_embed.append(nn.BatchNorm2d(196))
                in_embed.append(nn.Dropout(dropout_rates[0]))
                in_embed.append(nn.MaxPool2d(kernel_size=2, stride=2))

                in_embed.append( nn.Flatten())
                in_embed.append(nn.BatchNorm1d(36652))
                in_embed.append(nn.Linear(36652,1000))
                in_embed.append(nn.LeakyReLU()) 
                in_embed.append(nn.Linear(1000,hidden_sizes[0]))
            else:
                in_embed.append( nn.Flatten())
                in_embed.append(nn.BatchNorm1d(21632))
                in_embed.append(nn.Linear(21632,hidden_sizes[0]))
            self.in_embed = nn.Sequential(*in_embed)
        elif input_type == '3d':
            self.input_size = 85*85*85
            in_embed.append(nn.Conv3d(N_channels,8,kernel_size=3,stride=1,padding=1))
            in_embed.append(nn.LeakyReLU())
            in_embed.append(nn.BatchNorm3d(8))
            in_embed.append(nn.Dropout(dropout_rates[0]))
            in_embed.append(nn.MaxPool3d(kernel_size=2, stride=2))
            in_embed.append(nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=1))
            in_embed.append(nn.LeakyReLU())
            in_embed.append(nn.BatchNorm3d(16))
            in_embed.append(nn.Dropout(dropout_rates[1]))
            in_embed.append(nn.MaxPool3d(kernel_size=2, stride=2))
            in_embed.append(nn.Conv3d(16,32,kernel_size=3,stride=1,padding=1))
            in_embed.append(nn.LeakyReLU())
            in_embed.append(nn.BatchNorm3d(32))
            in_embed.append(nn.Dropout(dropout_rates[0]))
            in_embed.append(nn.MaxPool3d(kernel_size=2, stride=2))
            in_embed.append(nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1))
            in_embed.append(nn.LeakyReLU())
            in_embed.append(nn.BatchNorm3d(64))
            in_embed.append(nn.Dropout(dropout_rates[1]))
            in_embed.append(nn.MaxPool3d(kernel_size=2, stride=2))
            in_embed.append(nn.Conv3d(64,96,kernel_size=3,stride=1,padding=1))
            in_embed.append(nn.LeakyReLU())
            in_embed.append(nn.BatchNorm3d(96))
            in_embed.append(nn.Dropout(dropout_rates[0]))
            in_embed.append(nn.MaxPool3d(kernel_size=2, stride=2))
            in_embed.append(nn.Conv3d(96, 128, kernel_size=3, stride=1, padding=1))
            in_embed.append(nn.LeakyReLU())
            in_embed.append(nn.BatchNorm3d(128))
            in_embed.append(nn.Dropout(dropout_rates[1]))
            in_embed.append( nn.Flatten())
            in_embed.append(nn.BatchNorm1d(1024))
            in_embed.append(nn.Linear(1024,hidden_sizes[0]))

            self.in_embed = nn.Sequential(*in_embed)
        else:
            raise ValueError("input_type must be '1D', '2D', or '3D'")
        self.dir_embed = nn.Linear(2,hidden_sizes[0])
        layers = []
        in_size = hidden_sizes[0]*2
        for h in hidden_sizes[1:]:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.LeakyReLU())
            in_size = h
        output_size = (num_filters * 3 )*2  # fc, G for LS then Fc, G, fB for peak filters for each ears
        layers.append(nn.Linear(in_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x,direction):
        x=self.in_embed(x)
        dir_embed = self.dir_embed(direction)
        dir_embed = F.leaky_relu(dir_embed,0.01)
        x_dir= torch.cat((x, dir_embed), dim=1)
        return self.network(x_dir)


class ParametricIIRCascade(nn.Module):
    def __init__(self, num_filters, n_freqs=128, fs=44100):
        super().__init__()
        self.num_filters = num_filters
        self.fs = fs
        self.n_freqs = n_freqs
        f = torch.linspace(0, fs / 2, n_freqs)
        self.register_buffer("w", 2 * np.pi * f / fs)

    def forward(self, params):

        B = params.shape[0]
        params = params.view(B,self.num_filters,3)
        w = self.w.view(1, -1)  # (1, n_freqs)
        w1= w.repeat(B, 1)  # (B, n_freqs)
        w2=w.repeat(B,self.num_filters-2,1).view(B,self.num_filters-2,-1)  # (B, num_filters, n_freqs)

        fc, Q, G = params[:, 0,0], params[:,0, 1], params[:,0,2]
        fc = torch.sigmoid(fc) * (self.fs / 2)  # 0 to Nyquist
        Q = F.softplus(Q) + 1e-3  # Q > 0
        G=torch.sigmoid(G)*20-80  # Gain in [-80, 20]
        one = torch.ones_like(w1)
        ejw = torch.exp(-1j * w1)
        ejw2 = ejw ** 2   # first filter LFS (butterworth)
        omega_c= 2 * np.pi * fc / self.fs
        K=torch.tan(omega_c/2)
        den = 1 + np.sqrt(2) * K + K**2
        b0 = G * K**2 / den
        b1 = 2 * b0
        b2 = b0
        a1 = 2 * (K**2 - 1) / den
        a2 = (1 - np.sqrt(2) * K + K**2) / den
        b0=b0.repeat(1, self.n_freqs).view(B, -1)
        b1=b1.repeat(1, self.n_freqs).view(B, -1)
        b2=b2.repeat(1, self.n_freqs).view(B, -1)
        a1=a1.repeat(1, self.n_freqs).view(B, -1)
        a2=a2.repeat(1, self.n_freqs).view(B, -1)
        num_low_pass = b0*one + b1 * ejw + b2 * ejw2
        deno_low_pass = 1 + a1 * ejw + a2 * ejw2

        deno_low_pass=deno_low_pass.clone()
        deno_low_pass[deno_low_pass==0] = 1e-10
                # Avoid division by zero

        H_low_pass = torch.div(num_low_pass ,deno_low_pass)


        fc2, Q2, G2 = params[:, 1,0], params[:,1, 1], params[:,1,2]
        fc2 = torch.sigmoid(fc2) * (self.fs / 2)  # 0 to Nyquist
        Q2 = F.softplus(Q2) + 1e-3  # Q > 0
        G2=torch.sigmoid(G2)*20-80  # Gain in [-80, 20]
        omega_c = 2 * np.pi * fc2 / self.fs
        K = torch.tan(omega_c / 2)
        den = 1 + np.sqrt(2) * K + K**2
        b0 = G2 / den
        b1 = -2 * b0
        b2 = b0
        a1 = 2 * (K**2 - 1) / den
        a2 = (1 - np.sqrt(2) * K + K**2) / den

        b0=b0.repeat(1, self.n_freqs).view(B, -1)
        b1=b1.repeat(1, self.n_freqs).view(B, -1)
        b2=b2.repeat(1, self.n_freqs).view(B, -1)
        a1=a1.repeat(1, self.n_freqs).view(B, -1)
        a2=a2.repeat(1, self.n_freqs).view(B, -1)  

        num_high_pass = b0*one + b1 * ejw + b2 * ejw2
        deno_high_pass = 1 + a1 * ejw + a2 * ejw2

        deno_high_pass=deno_high_pass.clone()
        deno_high_pass[deno_high_pass==0] = 1e-10

        H_high_pass = torch.div(num_high_pass ,deno_high_pass)  # (B, n_freqs)


        fc3, Q3, G3 = params[:, 2:,0], params[:,2:, 1], params[:,2:,2]
        fc3 = torch.sigmoid(fc3) * (self.fs / 2)  # 0 to Nyquist
        Q3 = F.softplus(Q3) + 1e-3  # Q > 0
        G3=torch.sigmoid(G3)*40-20  # Gain in [-80, 20]
        A = 10**(G3 / 40)
        omega = 2 * np.pi * fc3 / self.fs
        alpha = torch.sin(omega) / (2 * Q3)
        cos_omega = torch.cos(omega)

        b0 = 1 + alpha * A
        b1 = -2 * cos_omega
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * cos_omega
        a2 = 1 - alpha / A
        b0,b1,b2 = b0 / a0, b1 / a0, b2 / a0
        a0,a1,a2= 1.0, a1 / a0, a2 / a0

        one_p = torch.ones_like(w2)
        ejw_p = torch.exp(-1j * w2)
        ejw2_p = ejw_p ** 2 

        b0=b0.repeat(1, self.n_freqs).view(B,self.num_filters-2, -1)
        b1=b1.repeat(1, self.n_freqs).view(B, self.num_filters-2,-1)
        b2=b2.repeat(1, self.n_freqs).view(B,self.num_filters-2, -1)
        a1=a1.repeat(1, self.n_freqs).view(B,self.num_filters-2, -1)
        a2=a2.repeat(1, self.n_freqs).view(B,self.num_filters-2, -1) 

        num_peak = b0*one_p + b1 * ejw_p + b2 * ejw2_p
        deno_peak = 1 + a1 * ejw_p + a2 * ejw2_p


        deno_peak=deno_peak.clone()
        deno_peak[deno_peak==0] = 1e-10
        # Avoid division by zero

        H_peak = torch.div(num_peak ,deno_peak)  # (B, num_filters-2, n_freqs)
        H_peak=20*torch.log10(torch.abs(H_peak)+0.0001)
        H_low_pass= 20*torch.log10(torch.abs(H_low_pass)+0.0001)
        H_high_pass= 20*torch.log10(torch.abs(H_high_pass)+0.0001)
        H_peak=torch.sum(H_peak,dim=1)
        H_total = H_low_pass + H_high_pass + H_peak  # (B, n_freqs)
        if(torch.isnan(H_total)).any():
            print("NaN detected in filter response")
            print(H_total)
            raise ValueError("NaN detected in filter response")

        log_mag =H_total.type(torch.float32)  # safer than log(mag + eps)
        return log_mag

class HRTFEstimator_Le_roux(nn.Module):
    def __init__(self, hidden_sizes, num_filters, n_freqs=128,input_type='1D',ear_input='left',dropout_rates=[0.2,0.2],distorted_photo=False):
        super(HRTFEstimator_Le_roux, self).__init__()
        self.mlp = MLP( hidden_sizes, num_filters,input_type,ear_input,dropout_rates=dropout_rates,distorted_photo=distorted_photo)
        self.iir_response_R = ParametricIIRCascade(num_filters, n_freqs)
        self.iir_response_L = ParametricIIRCascade(num_filters, n_freqs)
        self.num_filters = num_filters

    def forward(self, x, direction, out_HRTF=True):
        params = self.mlp(x,direction)  # output shape: (batch, num_filters * 3 * 2 ears)
        params_L = params[:, :self.num_filters * 3]
        params_R = params[:, self.num_filters * 3:]
        log_mag_R = self.iir_response_R(params_R)  # Right ear
        log_mag_L = self.iir_response_L(params_L)  # Left ear
        log_mag = torch.cat((log_mag_L, log_mag_R), dim=1)
        return log_mag
    
if __name__ == "__main__":
    model= HRTFEstimator_Le_roux(hidden_sizes=[256, 256, 128,64,32], num_filters=10, n_freqs=128,input_type='2d',ear_input='both',dropout_rates=[0.001, 0.001],distorted_photo=False)
    input_tensor = torch.randn(20,6,396,606)
    direction_input=torch.randn(20,2)
    output = model(input_tensor,direction_input)
    print(output.shape)