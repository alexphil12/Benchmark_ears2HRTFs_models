import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import io
from scipy.special import sph_harm
from tqdm import tqdm
import os

def sph_basis(N, theta, phi):
    # Calcul CPU via SciPy
    basis = []
    for n in range(N+1):
        for m in range(-n, n+1):
            basis.append(sph_harm(m, n, theta, phi))
    basis = np.array(basis).T  # (n_dirs, (N+1)^2)
    np.save(f'Spherical_based_save/spherical Base N={N}.npy', basis)

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
class Log_spectral_dis(nn.Module):
    def __init__(self):
        super(Log_spectral_dis, self).__init__()

    def forward(self, input, target):
        # Compute the loss
        eps=0.001
        loss = torch.sqrt(torch.mean(20*torch.log10(torch.abs((target+eps)/(input+eps)))))
        return loss
    
class MSE_root(nn.Module):
    def __init__(self):
        super(MSE_root, self).__init__()

    def forward(self, input, target):
        # Compute the loss
        loss = torch.sqrt(nn.MSELoss(reduction='mean')(input, target))
        return loss
    
def hrir_to_hrtf(hrir):
    # Convert HRIR to HRTF
    hrtf = 20*np.log10(np.abs(np.fft.fft(hrir, axis=-1))+0.0001)
    hrtf = hrtf[0:hrtf.shape[-1] // 2]
    return hrtf



def gen_image_global_evaluation_perf(std_ds,std_sf,mean_ds,mean_sf,input_type,epoch,model_name,sr=44100):
    nu=np.linspace(0,sr/2,len(mean_ds))
    fig,axarr=plt.subplots(1, 2,figsize=(20, 20))
    fig.suptitle("Global LSD evaluation at epochs "+str(epoch)+" for model "+model_name+ " and input type "+input_type)
    axarr[0].plot(nu,mean_ds,color="red", label="Mean LSD")
    axarr[0].fill_between(nu, mean_ds-std_ds, mean_ds+std_ds, color="red", alpha=0.3, label="±1 std")
    axarr[0].set_xscale("log")
    axarr[0].set_title('Mean LSD accross eval subjects and directions')
    axarr[0].set_xlabel('Frequency (Hz)')
    axarr[0].set_ylabel('Magnitude (dB)')
    axarr[0].legend()

    axarr[1].plot(mean_sf,color="blue", label="Mean LSD")
    axarr[1].fill_between(np.arange(len(mean_sf)), mean_sf-std_sf, mean_sf+std_sf, color="blue", alpha=0.3, label="±1 std")
    axarr[1].set_title('Mean LSD accross eval subjects and frequencies')
    axarr[1].set_xlabel('Direction index')
    axarr[1].set_ylabel('Magnitude (dB)')
    axarr[1].legend()
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    return fig

def gen_image_hrtf_evolve(hrtf_gt,hrtf_pred,incidence,epoch,model_name,input_type,sr=44100):
    nu=np.linspace(0,sr/2,hrtf_gt.shape[1]//2)
    
    fig,axarr=plt.subplots(2, 2,figsize=(20, 20))
    fig.suptitle("HRTF comparison of GT and Pred at epoch "+str(epoch)+" for model "+model_name+ " and input type "+input_type)
    axarr[0, 0].plot(nu, hrtf_gt[0,0:128], label='GT left')
    axarr[0, 0].plot(nu, hrtf_gt[0,128:], label='GT right')
    axarr[0, 0].plot(nu, hrtf_pred[0,0:128], label='Pred left')
    axarr[0, 0].plot(nu, hrtf_pred[0,128:], label='Pred right')
    LSD_left=np.square(hrtf_gt[0,0:128]-hrtf_pred[0,0:128]).mean(axis=None)
    LSD_right=np.square(hrtf_gt[0,128:]-hrtf_pred[0,128:]).mean(axis=None)
    LSD_left=np.sqrt(LSD_left)
    LSD_right=np.sqrt(LSD_right)
    axarr[0, 0].set_xscale("log")
    axarr[0, 0].set_title('Comparsion at azimuth '+str(incidence[0,0])+' and elevation '+str(incidence[0,1])+' with LSD left ear '+str(np.round(LSD_left,2))+' dB and right ear '+str(np.round(LSD_right,2))+' dB')
    axarr[0, 0].set_xlabel('Frequency (Hz)')
    axarr[0, 0].set_ylabel('Magnitude (dB)')
    axarr[0, 0].legend()
    axarr[0, 1].plot(nu, hrtf_gt[1,0:128], label='GT left')
    axarr[0, 1].plot(nu, hrtf_gt[1,128:], label='GT right')
    axarr[0, 1].plot(nu, hrtf_pred[1,0:128], label='Pred left')
    axarr[0, 1].plot(nu, hrtf_pred[1,128:], label='Pred right')
    LSD_left_1=np.square(hrtf_gt[1,0:128]-hrtf_pred[1,0:128]).mean(axis=None)
    LSD_right_1=np.square(hrtf_gt[1,128:]-hrtf_pred[1,128:]).mean(axis=None)
    LSD_left_1=np.sqrt(LSD_left_1)
    LSD_right_1=np.sqrt(LSD_right_1)
    axarr[0, 1].set_xscale("log")
    axarr[0, 1].set_title('Comparsion at azimuth '+str(incidence[1,0])+' and elevation '+str(incidence[1,1])+' with LSD left ear '+str(np.round(LSD_left_1,2))+' dB and right ear '+str(np.round(LSD_right_1,2))+' dB')
    axarr[0, 1].set_xlabel('Frequency (Hz)')
    axarr[0, 1].set_ylabel('Magnitude (dB)')
    axarr[0, 1].legend()
    axarr[1, 0].plot(nu, hrtf_gt[2,0:128], label='GT left')
    axarr[1, 0].plot(nu, hrtf_gt[2,128:], label='GT right')
    axarr[1, 0].plot(nu, hrtf_pred[2,0:128], label='Pred left')
    axarr[1, 0].plot(nu, hrtf_pred[2,128:], label='Pred right')
    LSD_left_2=np.square(hrtf_gt[2,0:128]-hrtf_pred[2,0:128]).mean(axis=None)
    LSD_right_2=np.square(hrtf_gt[2,128:]-hrtf_pred[2,128:]).mean(axis=None)
    LSD_left_2=np.sqrt(LSD_left_2)
    LSD_right_2=np.sqrt(LSD_right_2)
    axarr[1, 0].set_xscale("log")
    axarr[1, 0].set_title('Comparsion at azimuth '+str(incidence[2,0])+' and elevation '+str(incidence[2,1]) +' with LSD left ear '+str(np.round(LSD_left_2,2))+' dB and right ear '+str(np.round(LSD_right_2,2))+' dB')
    axarr[1, 0].set_xlabel('Frequency (Hz)')
    axarr[1, 0].set_ylabel('Magnitude (dB)')
    axarr[1,0].legend()
    axarr[1, 1].plot(nu, hrtf_gt[3,0:128], label='GT left')
    axarr[1, 1].plot(nu, hrtf_gt[3,128:], label='GT left')
    axarr[1, 1].plot(nu, hrtf_pred[3,0:128], label='Pred left')
    axarr[1, 1].plot(nu, hrtf_pred[3,128:], label='Pred right')
    LSD_left_3=np.square(hrtf_gt[3,0:128]-hrtf_pred[3,0:128]).mean(axis=None)
    LSD_right_3=np.square(hrtf_gt[3,128:]-hrtf_pred[3,128:]).mean(axis=None)
    LSD_left_3=np.sqrt(LSD_left_3)
    LSD_right_3=np.sqrt(LSD_right_3)
    axarr[1, 1].set_xscale("log")
    axarr[1, 1].set_title('Comparsion at azimuth '+str(incidence[3,0])+' and elevation '+str(incidence[3,1])+' with LSD left ear '+str(np.round(LSD_left_3,2))+' dB and right ear '+str(np.round(LSD_right_3,2))+' dB')
    axarr[1, 1].set_xlabel('Frequency (Hz)')
    axarr[1, 1].set_ylabel('Magnitude (dB)')
    axarr[1,1].legend()

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    return fig

def Calculate_SHT_coef(HRTF_y_set_L,HRTF_y_set_R,Y_basis,alpha=0,device='cpu'):
    YH = torch.conj(Y_basis).T
    YH = YH.type(torch.complex64)
    A = YH @ Y_basis
    if alpha > 0:
        A += alpha * torch.eye(A.shape[0], dtype=torch.complex64, device=device)

    B_l = YH @ HRTF_y_set_L
    B_r = YH @ HRTF_y_set_R  # ( (N+1)^2, n_freqs )

    # Solve AX = B
    coeffs_left = torch.linalg.solve(A, B_l)
    coeffs_right = torch.linalg.solve(A, B_r)
    coeffs_left = coeffs_left.type(torch.complex64)
    coeffs_right = coeffs_right.type(torch.complex64)
    coeffs_left = coeffs_left.unsqueeze(dim=1)
    coeffs_right = coeffs_right.unsqueeze(dim=1)
    SHT_coef = torch.stack((coeffs_left, coeffs_right), dim=1)  # ( (N+1)^2, 2, n_freqs )
    SHT_coef =  SHT_coef.detach().cpu().numpy()
    return SHT_coef

def Calculate_inverse_grid(SHT_coef,Y_basis):
    if SHT_coef.ndim==4:
        N= int(SHT_coef.shape[0])
        HRTF_reconstructed=torch.zeros((N,793,2),dtype=torch.float32)
        for i in range(N):
            SHT_coef_left = SHT_coef[i,:, 0, :]
            SHT_coef_right = SHT_coef[i,:, 1,:]
            recon_measured = Y_basis @ SHT_coef_left
            recon_measured_right = Y_basis @ SHT_coef_right  # ( n_points, n_freqs )
            HRTF_reconstructed_inter = torch.stack((recon_measured, recon_measured_right), dim=1)
            HRTF_reconstructed_inter= HRTF_reconstructed_inter.squeeze(2) # ( n_points, 2, n_freqs )
            real=torch.view_as_real(HRTF_reconstructed_inter)  
            HRTF_reconstructed[i,:,:]=real[i,:,0].type(torch.float32)
        HRTF_reconstructed=HRTF_reconstructed.detach().cpu().numpy()
    else:
        SHT_coef_left = SHT_coef[:, 0, :]
        SHT_coef_right = SHT_coef[:, 1,:]  # ( (N+1)^2, n_freqs )
        recon_measured = Y_basis @ SHT_coef_left
        recon_measured_right = Y_basis @ SHT_coef_right  # ( n_points, n_freqs )
        HRTF_reconstructed = torch.stack((recon_measured, recon_measured_right), dim=1)  # ( n_points, 2, n_freqs )
        real=torch.view_as_real(HRTF_reconstructed)
        HRTF_reconstructed=real[:,:,:,0].type(torch.float32)
        HRTF_reconstructed=HRTF_reconstructed.detach().cpu().numpy()
    return HRTF_reconstructed
    



class EarlyStopping:
    def __init__(self, patience=5, delta=0,verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)
    
def training_loop(num_epochs, incidence_data, DataLoader_train, DataLoader_test, model, criterion, optimizer, device, output_type, out_HRTF, nu, early_stoping, path_model_weight, run,fold_idx,batch_size,valid_batch_size):
    global_min_eval = float('inf')
    eval_losses = []
    train_losses = []
    run.define_metric("train/*",step_metric="train/step")
    if early_stoping:
        early_stop=EarlyStopping(verbose=True, patience=10,delta=0.001)
    for epoch in range(num_epochs):
        index_inci=list(range(len(incidence_data)))
        model.train()
        running_loss = 0.0
        running_LSD=0.0
        loop = tqdm(DataLoader_train, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for inputs, labels in loop:
            X_data,y_labels = inputs.to(device) ,labels.to(device)
            perm=np.random.permutation(index_inci)
            perm_nu=np.random.permutation(range(len(nu)))
            # Forward pass
            if "dir" in output_type:
                for i in range(len(incidence_data)):
                    inci_gpu=incidence_data.to(device)
                    inci_gpu=inci_gpu[perm[i],:]
                    inci_gpu=inci_gpu.repeat((X_data.shape[0],1))/360
                    outputs = model(X_data,inci_gpu,out_HRTF=out_HRTF)
                    loss = criterion(outputs, y_labels[:,perm[i],:,:].reshape_as(outputs))
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1000)
                    optimizer.step()
                    running_loss += loss.item()
            elif output_type=="SHT_set":
                Out_eval=torch.zeros_like(y_labels)
                Out_eval=Out_eval.to(device)
                for i in range(len(nu)):
                    freq= torch.ones((X_data.shape[0],1),device=device)*perm_nu[i]
                    freq=freq.type(torch.long)   
                    outputs = model(X_data,freq,out_HRTF=out_HRTF)
                    Out_eval[:,:,:,perm_nu[i]]=outputs
                    if out_HRTF==False:
                        loss = criterion(torch.view_as_real(outputs),torch.view_as_real(y_labels[:,:,:,:,perm_nu[i]].reshape_as(outputs)))
                    elif out_HRTF==True:
                        loss = criterion(outputs, y_labels[:,:,:,perm_nu[i]].reshape_as(outputs))
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1000)
                    optimizer.step()
                    running_loss += loss.item() 
                diff=(Out_eval - y_labels)**2
                bef_sqrt=torch.sqrt(diff.mean(dim=3))
                LSD=bef_sqrt.mean([1,2]).sum().item()
                running_LSD+=LSD                
            else:
                outputs = model(X_data,out_HRTF=out_HRTF)
                outputs=outputs.to(device)
                loss = criterion(outputs, y_labels.reshape_as(outputs))
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1000)
                optimizer.step()
                running_loss += loss.item()

            loop.set_postfix(loss=running_loss)
        if "dir" in output_type:
            epoch_loss = running_loss / (len(DataLoader_train)*len(incidence_data))
        elif output_type=="SHT_set":
            epoch_loss = running_LSD / (len(DataLoader_train)*batch_size)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        model.eval()
        test_loss=0.0
        with torch.no_grad():
            torch.cuda.empty_cache()
            loop_eval = tqdm(DataLoader_test, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
            running_loss_eval = 0.0
            running_LSD_ev=0.0
            for inputs, labels in loop_eval:
                X_data_ev, y_labels_ev = inputs.to(device) ,labels.to(device)
                perm=np.random.permutation(index_inci)
                if "dir" in output_type:
                    for i in range(len(incidence_data)):
                        X_data_ev
                        inci_gpu=incidence_data.to(device)
                        inci_gpu=inci_gpu[perm[i],:]
                        inci_gpu=inci_gpu.repeat((X_data_ev.shape[0],1))/360
                        outputs = model(X_data_ev,inci_gpu,out_HRTF=out_HRTF)
                        loss = criterion(outputs, y_labels_ev[:,perm[i],:,:].reshape_as(outputs))
                        running_loss_eval += loss.item()
                elif output_type=="SHT_set":
                    Out_eval=torch.zeros_like(y_labels_ev)
                    Out_eval=Out_eval.to(device)
                    for i in range(len(nu)):
                        freq= torch.ones((X_data_ev.shape[0],1),device=device)*perm_nu[i]
                        freq=freq.type(torch.long)   
                        outputs = model(X_data_ev,freq,out_HRTF=out_HRTF)
                        Out_eval[:,:,:,perm_nu[i]]=outputs
                        if out_HRTF==False:
                            loss = criterion(torch.view_as_real(outputs),torch.view_as_real(y_labels_ev[:,:,:,:,perm_nu[i]].reshape_as(outputs)))
                        elif out_HRTF==True:
                            loss = criterion(outputs, y_labels_ev[:,:,:,perm_nu[i]].reshape_as(outputs))
                        running_loss_eval += loss.item()
                    diff=(Out_eval - y_labels_ev)**2
                    bef_sqrt=torch.sqrt(diff.mean(dim=3))
                    LSD=bef_sqrt.mean([1,2]).sum().item()
                    running_LSD_ev+=LSD
                else:
                    outputs = model(X_data_ev,out_HRTF=out_HRTF)
                    outputs=outputs.to(device)
                    loss = criterion(outputs, y_labels_ev.reshape_as(outputs))
                    running_loss_eval += loss.item()
                loop_eval.set_postfix(loss=running_loss_eval)
            if "dir" in output_type:
                test_loss = running_loss_eval / (len(DataLoader_test)*len(incidence_data))
            elif output_type=="SHT_set":
                test_loss = running_LSD_ev / (len(DataLoader_test)*valid_batch_size)
            log_dict={"train/step":epoch,
                      f"train/train_loss_fold_{fold_idx+1}":epoch_loss,
                      f"train/eval_loss_fold_{fold_idx+1}":test_loss
                      }
            run.log(log_dict)
            eval_losses.append(test_loss)
            train_losses.append(epoch_loss)
            if test_loss < global_min_eval:
                global_min_eval = test_loss
                torch.save(model, os.path.join(path_model_weight, f"best_model_{epoch+1}_with_loss_{test_loss}.pth"))
                print(f"Model saved at epoch {epoch+1} with test loss {test_loss:.4f}")
            print(f"Epoch {epoch+1}/{num_epochs}, test Loss: {test_loss:.4f}")
        # if early_stoping:
        #     early_stop(test_loss, model)
        #     if early_stop.early_stop:
        #         if early_stop.verbose:
        #             print(f"Early stopping at epoch {epoch+1}")
        #         break
    run.log({f"best_test_loss_fold_{fold_idx+1}":global_min_eval})
    return train_losses, eval_losses, global_min_eval
        
if __name__ == "__main__":
    N=17
    incidence = torch.load('incidence.pt')
    theta = torch.deg2rad(incidence[:, 0]).detach().numpy()
    phi = torch.deg2rad(90 - incidence[:, 1]).detach().numpy()
    sph_basis(N, theta, phi)