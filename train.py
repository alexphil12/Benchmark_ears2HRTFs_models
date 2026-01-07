"""
Train a neural network model for HRTF (Head-Related Transfer Function) estimation.
Args:
    args (Namespace): A namespace object containing various training parameters including:
        - hidden_sizes (list): Sizes of hidden layers in the model (for Le-roux architecture).
        - exp_name (str): Name of the experiment.
        - dropout (float): Dropout rate for the model (for Woo-lee architecture).
        - ears_dataset_path (str): Path to the dataset containing ear measurements.
        - HRTF_dataset_path (str): Path to the dataset containing HRTF data.
        - total_epochs (int): Total number of training epochs.
        - batch_size (int): Batch size for training.
        - lr (float): Learning rate for the optimizer.
        - model (str): Name of the model to be used ('Woo-lee', 'Le-roux', or 'Manlin-Zhao').
        - early_stop (str): Flag to indicate if early stopping is enabled.
        - valid_batch_size (int): Batch size for validation.
        - ear_input (str): Type of ear input data (left, right or both).
        - L_order_SHT (int): Order of spherical harmonics decomposition(for Manlin-Zhao model).
        - N_Filters_le_roux (int): Number of filters (for the Le-roux model).
        - out_HRTF (str): Flag to indicate if outputting HRTF is enabled(usefull for Manlin-Zhao and Woo-lee models).
        - disto_photo (str): Flag to indicate if distorted photo input is used(Only used when input type=2d).
Returns:
    None: The function does not return any value but logs training, evaluation metrics and graphs to WandB.
"""
import torch
from torch import nn, optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from Datasets import HRTF_mesh_dataset
import os
import wandb
import numpy as np
from utils import hrir_to_hrtf,gen_image_hrtf_evolve,Calculate_inverse_grid,EarlyStopping,gen_image_global_evaluation_perf,MSE_root,init_weights
import matplotlib.pyplot as plt
import faulthandler
faulthandler.enable()

def train(args):
    # Extract training hyperparameters from args
    hidden_sizes=args.hidden_sizes
    Name_exp=args.exp_name
    dropout=args.dropout
    Train_mode=args.train
    Test_mode=args.evaluate
    ears_dataset_path=args.ears_dataset_path  
    HRTF_dataset_path=args.HRTF_dataset_path
    num_epochs=args.total_epochs
    batch_size=args.batch_size
    lr=args.lr
    k_folds=args.k_folds
    test_fold=args.test_fold
    model_name=args.model
    early_stoping=args.early_stop
    
    # Convert string flags to boolean values
    if early_stoping=="True":
        early_stoping=True
    else:
        early_stoping=False
    valid_batch_size=args.valid_batch_size
    ear_input=args.ear_input  
    L_order_SHT=args.L_order_SHT
    percent_of_data=args.percent_of_data
    Number_of_filters=args.N_Filters_le_roux
    out_HRTF=args.out_HRTF
    
    # Convert output HRTF flag
    if out_HRTF=="True":
        out_HRTF=True
    else:
        out_HRTF=False
    distorted_photo=args.disto_photo
    
    # Convert distorted photo flag
    if distorted_photo=="True":
        distorted_photo=True
    else:
        distorted_photo=False
    bool_forward=False
    
    # Set device to GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load spherical harmonics basis functions for Manlin-Zhao model
    Y_basis=torch.tensor(np.load("/home/PhilipponA/Script_model_HRTF_Sonicom/Spherical_based_save/spherical Base N="+str(L_order_SHT)+".npy"),dtype=torch.complex64)
    Y_basis=Y_basis.to(device)  
    
    # Determine output type based on model architecture
    if model_name=="Woo-lee":
        output_type="IR_dir"
    elif model_name=="Le-roux":
        output_type="HRTF_dir"
    elif model_name=="Manlin-Zhao":
        output_type="SHT_set"     
    else:
        raise ValueError("Invalid model name. Choose 'Woo-lee', 'Le-roux', or 'Manlin-Zhao'.")
    
    # Determine input data type
    input_type=args.input_type
    if input_type=="1d":
        whole_input="ears_measurements"
    elif input_type=="2d":
        whole_input="ears_photo"
    elif input_type=="3d":
        whole_input="Ears_3D_voxels"

    # Create experiment name and directories for logging
    whole_xp_name="/"+Name_exp+"_model_"+model_name
    if not os.path.exists(args.log_tensorboard_dir+whole_xp_name):
        os.makedirs(args.log_tensorboard_dir+whole_xp_name+"/"+input_type)
    path_tensorboard = args.log_tensorboard_dir+whole_xp_name+"/"+input_type
    
    # Create directories for model weights
    if not os.path.exists(args.Model_dir+whole_xp_name+"_"+input_type):
        os.makedirs(args.Model_dir+whole_xp_name+"_"+input_type)
    path_model_weight = args.Model_dir+whole_xp_name+"_"+input_type

    from pathlib import Path

    # Load WandB API key from file
    file_path = Path('clef_wandb.txt')
    wan_key = file_path.read_text()

    # Initialize WandB for experiment logging
    wandb.login(key=wan_key)
    disto_label="distorted_photo" if distorted_photo else "no_distorted_photo"
    run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="Enter_your_entity_name",
    project="Enter_your_project_name",
    config={"learning_rate": lr,"architecture":model_name ,"dataset": "Sonicom",
        "epochs": num_epochs,"batch_size": batch_size,"input_type": input_type,
        "output_type": output_type,"ear_input": ear_input,"N_filters_le_roux": Number_of_filters,"SHT_decomposition_order":L_order_SHT,"type_of_2D_input":disto_label}
)
    
    # Create frequency array for frequency-dependent analysis
    nu=np.linspace(0,44100/2,128)
    
    # Initialize early stopping if enabled
    if early_stoping:
        early_stop=EarlyStopping(verbose=True, patience=10,delta=0.001)

    # Create training and validation datasets
    dataset_train=HRTF_mesh_dataset(ears_dataset_path=ears_dataset_path,HRTF_dataset_path=HRTF_dataset_path,type_of_data=input_type,output_type=output_type,Train_data=True,L=L_order_SHT,mode=ear_input,distorted_photo=distorted_photo,percent_of_data=percent_of_data,Y_basis=Y_basis,device=device)
    dataset_test=HRTF_mesh_dataset(ears_dataset_path=ears_dataset_path,HRTF_dataset_path=HRTF_dataset_path,type_of_data=input_type,output_type=output_type,Train_data=False,L=L_order_SHT,mode=ear_input,distorted_photo=distorted_photo,percent_of_data=percent_of_data,Y_basis=Y_basis,device=device)
    dataset_val=HRTF_mesh_dataset(ears_dataset_path=ears_dataset_path,HRTF_dataset_path=HRTF_dataset_path,type_of_data=input_type,output_type=output_type,Train_data=False,Test_data=False,L=L_order_SHT,mode=ear_input,distorted_photo=distorted_photo,percent_of_data=percent_of_data,Y_basis=Y_basis,device=device)

    # Create datasets with HRTF output format for models that support it
    dataset_train_out_HRTF=HRTF_mesh_dataset(ears_dataset_path=ears_dataset_path,HRTF_dataset_path=HRTF_dataset_path,type_of_data=input_type,output_type=output_type,Train_data=True,L=L_order_SHT,mode=ear_input,out_HRTF=out_HRTF,distorted_photo=distorted_photo,percent_of_data=percent_of_data,Y_basis=Y_basis,device=device)
    dataset_test_out_HRTF=HRTF_mesh_dataset(ears_dataset_path=ears_dataset_path,HRTF_dataset_path=HRTF_dataset_path,type_of_data=input_type,output_type=output_type,Train_data=False,L=L_order_SHT,mode=ear_input,out_HRTF=out_HRTF,distorted_photo=distorted_photo,percent_of_data=percent_of_data,Y_basis=Y_basis,device=device)
    dataset_val_out_HRTF=HRTF_mesh_dataset(ears_dataset_path=ears_dataset_path,HRTF_dataset_path=HRTF_dataset_path,type_of_data=input_type,output_type=output_type,Train_data=False,Test_data=False,L=L_order_SHT,mode=ear_input,out_HRTF=out_HRTF,distorted_photo=distorted_photo,percent_of_data=percent_of_data,Y_basis=Y_basis,device=device)
    
    # Select appropriate dataset based on out_HRTF flag
    if out_HRTF:
        DataLoader_train = DataLoader(dataset_train_out_HRTF, batch_size=batch_size, shuffle=True)
        DataLoader_test = DataLoader(dataset_test_out_HRTF, batch_size=valid_batch_size, shuffle=False)
    else:
        DataLoader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        DataLoader_test = DataLoader(dataset_test, batch_size=valid_batch_size, shuffle=False)

    # Extract incidence directions from training dataset
    incidence_data=dataset_train.incidence_dir
    
    # Instantiate model based on selected architecture
    if model_name=="Woo-lee":
        from models.Woo_Lee_simple_model import Woo_lee_Model
        model = Woo_lee_Model(input_type=input_type,ear_input=ear_input,dropout_rates=dropout,distorted_photo=distorted_photo)
    elif model_name=="Le-roux":
        from models.Le_roux_model_ddsp import HRTFEstimator_Le_roux
        model = HRTFEstimator_Le_roux(hidden_sizes=hidden_sizes, num_filters=Number_of_filters, n_freqs=128,input_type=input_type,ear_input=ear_input,dropout_rates=dropout,distorted_photo=distorted_photo)
    elif model_name=="Manlin-Zhao":
        from models.Manlin_Zhao_model_sht import Manlin_Zhao_Model
        model = Manlin_Zhao_Model(input_type=input_type,order_of_sht=L_order_SHT,num_of_frequency=128,ear_input=ear_input,incidence=incidence_data,distorted_photo=distorted_photo,Y_basis=Y_basis)
    
    # Initialize model weights and move to device
    model.apply(init_weights)
    model = model.to(device)

    # Define loss function and optimizer
    criterion = MSE_root()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    global_min_eval=1e10
    
    # Watch model with WandB
    wandb.watch(model)
    
    # Training loop over epochs
    for epoch in range(num_epochs):
        index_inci=list(range(len(incidence_data)))
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_LSD=0.0
        loop = tqdm(DataLoader_train, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        
        for inputs, labels in loop:
            X_data,y_labels = inputs.to(device) ,labels.to(device)
            perm=np.random.permutation(index_inci)
            perm_nu=np.random.permutation(range(len(nu)))
            
            # Forward pass with different handling for directional vs SHT output types
            if "dir" in output_type:
                # For directional output models (Woo-lee, Le-roux)
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
                # For spherical harmonics decomposition output (Manlin-Zhao)
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
                # Calculate Log Spectral Distance (LSD) for SHT output
                diff=(Out_eval - y_labels)**2
                bef_sqrt=torch.sqrt(diff.mean(dim=3))
                LSD=bef_sqrt.mean([1,2]).sum().item()
                running_LSD+=LSD
                                 
            else:
                # For direct HRTF output models
                outputs = model(X_data,out_HRTF=out_HRTF)
                outputs=outputs.to(device)
                loss = criterion(outputs, y_labels.reshape_as(outputs))
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1000)
                optimizer.step()
                running_loss += loss.item()

            loop.set_postfix(loss=running_loss)
        
        # Calculate epoch loss
        if "dir" in output_type:
            epoch_loss = running_loss / (len(DataLoader_train)*len(incidence_data))
        elif output_type=="SHT_set":
            epoch_loss = running_LSD / (len(DataLoader_train)*batch_size)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        
        # Evaluation phase
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
                    # Evaluate directional models
                    for i in range(len(incidence_data)):
                        X_data_ev
                        inci_gpu=incidence_data.to(device)
                        inci_gpu=inci_gpu[perm[i],:]
                        inci_gpu=inci_gpu.repeat((X_data_ev.shape[0],1))/360
                        outputs = model(X_data_ev,inci_gpu,out_HRTF=out_HRTF)
                        loss = criterion(outputs, y_labels_ev[:,perm[i],:,:].reshape_as(outputs))
                        running_loss_eval += loss.item()
                elif output_type=="SHT_set":
                    # Evaluate spherical harmonics models
                    Out_eval=torch.zeros_like(y_labels_ev)
                    Out_eval=Out_eval.to(device)
                    for i in range(len(nu)):
                        freq= torch.ones((X_data_ev.shape[0],1),device=device)*perm_nu[i]
                        freq=freq.type(torch.long)   
                        outputs = model(X_data_ev,freq,out_HRTF=out_HRTF)
                        Out_eval[:,:,:,:,perm_nu[i]]=outputs
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
                    # Evaluate direct HRTF models
                    outputs = model(X_data_ev,out_HRTF=out_HRTF)
                    outputs=outputs.to(device)
                    loss = criterion(outputs, y_labels_ev.reshape_as(outputs))
                    running_loss_eval += loss.item()
                loop_eval.set_postfix(loss=running_loss_eval)
            
            # Calculate test loss and save best model
            if "dir" in output_type:
                test_loss = running_loss_eval / (len(DataLoader_test)*len(incidence_data))
            elif output_type=="SHT_set":
                test_loss = running_LSD_ev/ (len(DataLoader_test)*valid_batch_size)
            run.log({"test-loss":test_loss,
                     "train-loss":epoch_loss})
            if test_loss < global_min_eval:
                global_min_eval = test_loss
                torch.save(model, os.path.join(path_model_weight, f"best_model_{epoch+1}_with_loss_{test_loss}.pth"))
                print(f"Model saved at epoch {epoch+1} with test loss {test_loss:.4f}")
            print(f"Epoch {epoch+1}/{num_epochs}, test Loss: {test_loss:.4f}")
        
        # Early stopping check
        if early_stoping:
            early_stop(test_loss, model)
            if early_stop.early_stop:
                if early_stop.verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
        
        # === VISUALIZATION PART ===
        # Generate evaluation plots every 3 epochs
        if epoch % 3 == 0:
            if model_name=="Woo-lee":
                # === Woo-lee Model Visualization ===
                # Load sample data from validation and training sets
                x_data,IR=dataset_val.__getitem__(0)
                x_data_train,IR_train=dataset_train.__getitem__(0)
                x_data=x_data.to(device)
                x_data_train=x_data_train.to(device)
                inci_gpu=incidence_data.to(device)
                
                # Select specific incidence angles for visualization
                start=5
                index_inci=[start,start+11*15,start+11*15*2,start+11*15*3]
                outputs=[]
                outputs_train=[]
                
                # Initialize arrays for storing ground truth and predicted HRTF
                HRTF_GT=np.zeros((len(index_inci),256))
                HRTF_pred=np.zeros((len(index_inci),256))
                HRTF_GT_train=np.zeros((len(index_inci),256))
                HRTF_pred_train=np.zeros((len(index_inci),256))
                
                # Forward pass for selected incidence angles
                for i in range(len(index_inci)):
                    index_read= inci_gpu[index_inci[i],:]/360
                    out = model(x_data.unsqueeze(0),index_read.unsqueeze(0))
                    out_train = model(x_data_train.unsqueeze(0),index_read.unsqueeze(0))
                    outputs.append(out.cpu().detach().numpy())
                    outputs_train.append(out_train.cpu().detach().numpy())
                
                inci_plot=[]
                inci_plot_train=[]
                
                # Process predictions to extract HRTF values
                for i in range(len(index_inci)):
                    # Extract ground truth HRTF (convert HRIR to HRTF)
                    HRTF_GT[i,0:128]=hrir_to_hrtf(IR[index_inci[i],0,:])
                    HRTF_GT[i,128:256]=hrir_to_hrtf(IR[index_inci[i],1,:])
                    HRTF_GT_train[i,0:128]=hrir_to_hrtf(IR_train[index_inci[i],0,:])
                    HRTF_GT_train[i,128:256]=hrir_to_hrtf(IR_train[index_inci[i],1,:])
                    
                    # Extract predicted HRTF (left and right channels)
                    hrir_l=np.squeeze(outputs[i])
                    hrir_l=hrir_l[0:256]
                    hrir_r=np.squeeze(outputs[i])
                    hrir_r=hrir_r[256:]
                    hrir_l_train=np.squeeze(outputs_train[i])
                    hrir_l_train=hrir_l_train[0:256]
                    hrir_r_train=np.squeeze(outputs_train[i])
                    hrir_r_train=hrir_r_train[256:]
                    
                    # Convert predicted HRIR to HRTF
                    HRTF_pred[i,0:128]=hrir_to_hrtf(hrir_l)
                    HRTF_pred[i,128:256]=hrir_to_hrtf(hrir_r)
                    HRTF_pred_train[i,0:128]=hrir_to_hrtf(hrir_l_train)
                    HRTF_pred_train[i,128:256]=hrir_to_hrtf(hrir_r_train)
                    
                    # Store incidence angles for plotting
                    inci_plot.append(incidence_data[index_inci[i],0:2].detach().numpy())
                    inci_plot_train.append(incidence_data[index_inci[i],0:2].detach().numpy())
                
                inci_plot=np.array(inci_plot)
                inci_plot_train=np.array(inci_plot_train)
                
                # Generate HRTF evolution plots
                fig=gen_image_hrtf_evolve(HRTF_GT,HRTF_pred,inci_plot,epoch+1,model_name,input_type)
                fig_train=gen_image_hrtf_evolve(HRTF_GT_train,HRTF_pred_train,inci_plot_train,epoch+1,model_name,input_type)
                run.log({"HRTF_evolution_val_data": wandb.Image(fig)})
                run.log({"HRTF_evolution_train_data": wandb.Image(fig_train)})

                # === Global evaluation on entire validation set ===
                N=dataset_val.__len__()
                pred_HRTF=np.zeros((N,793,256))
                GT_HRTF=np.zeros((N,793,256))
                shape=torch.tensor(x_data.shape)
                out_dim=torch.cat((torch.tensor(N).unsqueeze(0),shape),dim=0)
                X_data=torch.zeros(list(out_dim))
                
                # Collect all validation data
                for j in range(N):
                    x_data,HRTF_gt=dataset_val.__getitem__(j,out_HRTF=True)
                    inter=HRTF_gt.detach().numpy()
                    GT_HRTF[j,:,0:128]=inter[:,0,:]
                    GT_HRTF[j,:,128:256]=inter[:,1,:]
                    X_data[j,:]=x_data
                
                inci_gpu=incidence_data.to(device)
                X_data=X_data.to(device)         
                
                # Forward pass for all incidence directions
                for k in range(793):
                    inci_gpu_in=inci_gpu[k,:].repeat((X_data.shape[0],1))/360
                    out=model(X_data,inci_gpu_in,True)
                    inter=out.cpu().detach().numpy()
                    pred_HRTF[:,k,:]=inter[0,:]
                diff=(GT_HRTF-pred_HRTF)**2
                
            if model_name=="Le-roux":
                # === Le-roux Model Visualization ===
                # Load sample data
                x_data,IR=dataset_val.__getitem__(0)
                x_data_train,IR_train=dataset_train.__getitem__(0)
                x_data=x_data.to(device)
                x_data_train=x_data_train.to(device)
                inci_gpu=incidence_data.to(device)
                
                # Select specific incidence angles
                start=5
                index_inci=[start,start+11*15,start+11*15*2,start+11*15*3]
                outputs=[]
                outputs_train=[]
                HRTF_GT=np.zeros((len(index_inci),256))
                HRTF_pred=np.zeros((len(index_inci),256))
                HRTF_GT_train=np.zeros((len(index_inci),256))
                HRTF_pred_train=np.zeros((len(index_inci),256))
                
                # Forward pass and collect outputs
                for i in range(len(index_inci)):
                    index_read= inci_gpu[index_inci[i],:]/360
                    out = model(x_data.unsqueeze(0),index_read.unsqueeze(0))
                    out_train = model(x_data_train.unsqueeze(0),index_read.unsqueeze(0))
                    outputs.append(out.cpu().detach().numpy())
                    outputs_train.append(out_train.cpu().detach().numpy())
                
                inci_plot=[]
                inci_plot_train=[]
                
                # Process predictions
                for i in range(len(index_inci)):
                    # Le-roux outputs HRTF directly, not HRIR
                    HRTF_GT[i,0:128]=IR[index_inci[i],0,:]
                    HRTF_GT[i,128:256]=IR[index_inci[i],1,:]
                    HRTF_GT_train[i,0:128]=IR_train[index_inci[i],0,:]
                    HRTF_GT_train[i,128:256]=IR_train[index_inci[i],1,:]
                    
                    hrir_l=np.squeeze(outputs[i])
                    hrir_l=hrir_l[0:128]
                    hrir_r=np.squeeze(outputs[i])
                    hrir_r=hrir_r[128:]
                    hrir_l_train=np.squeeze(outputs_train[i])
                    hrir_l_train=hrir_l_train[0:128]
                    hrir_r_train=np.squeeze(outputs_train[i])
                    hrir_r_train=hrir_r_train[128:]
                    
                    HRTF_pred[i,0:128]=hrir_l
                    HRTF_pred[i,128:256]=hrir_r
                    HRTF_pred_train[i,0:128]=hrir_l_train
                    HRTF_pred_train[i,128:256]=hrir_r_train
                    
                    inci_plot.append(incidence_data[index_inci[i],0:2].detach().numpy())
                    inci_plot_train.append(incidence_data[index_inci[i],0:2].detach().numpy())
                
                inci_plot=np.array(inci_plot)
                inci_plot_train=np.array(inci_plot_train)
                
                # Generate and log plots
                fig=gen_image_hrtf_evolve(HRTF_GT,HRTF_pred,inci_plot,epoch+1,model_name,input_type)
                fig_train=gen_image_hrtf_evolve(HRTF_GT_train,HRTF_pred_train,inci_plot_train,epoch+1,model_name,input_type)
                run.log({"HRTF_evolution_val_data": wandb.Image(fig)})
                run.log({"HRTF_evolution_train_data": wandb.Image(fig_train)})

                # === Global evaluation ===
                N=dataset_val.__len__()
                pred_HRTF=np.zeros((N,793,256))
                GT_HRTF=np.zeros((N,793,256))
                shape=torch.tensor(x_data.shape)
                out_dim=torch.cat((torch.tensor(N).unsqueeze(0),shape),dim=0)
                X_data=torch.zeros(list(out_dim))
                
                # Collect validation data
                for j in range(N):
                    x_data,HRTF_gt=dataset_val.__getitem__(j,True)
                    inter=HRTF_gt.detach().numpy()
                    GT_HRTF[j,:,0:128]=inter[:,0,:]
                    GT_HRTF[j,:,128:256]=inter[:,1,:]
                    X_data[j,:]=x_data
                
                inci_gpu=incidence_data.to(device)
                X_data=X_data.to(device)         
                
                # Evaluate on all incidence directions
                for k in range(793):
                    inci_gpu_in=inci_gpu[k,:].repeat((X_data.shape[0],1))/360
                    out=model(X_data,inci_gpu_in)
                    inter=out.cpu().detach().numpy()
                    pred_HRTF[:,k,:]=inter[0,:]
                diff=(GT_HRTF-pred_HRTF)**2
                
            if model_name=="Manlin-Zhao":
                # === Manlin-Zhao Model Visualization ===
                if bool_forward==False:
                    # Load sample data
                    x_data,coefs_lab=dataset_val.__getitem__(0)
                    x_data_train,coefs_lab_train=dataset_train.__getitem__(0)
                    nu=list(range(128))
                    nu=torch.tensor(nu,dtype=torch.long).to(device) 
                    coefs_lab=coefs_lab.to(device)
                    coefs_lab_train=coefs_lab_train.to(device)
                    coefs_lab=coefs_lab.squeeze(2) 
                    coefs_lab_train=coefs_lab_train.squeeze(2)
                    
                    # Convert spherical harmonics coefficients to spatial grid
                    grids_gt=Calculate_inverse_grid(coefs_lab,Y_basis=Y_basis)
                    grids_gt_train=Calculate_inverse_grid(coefs_lab_train,Y_basis=Y_basis)
                    
                    x_data=x_data.to(device)
                    x_data_train=x_data_train.to(device)
                    inci_gpu=incidence_data.to(device)
                    
                    # Select incidence angles for visualization
                    start=5
                    index_inci=[start,start+11*15,start+11*15*2,start+11*15*3]
                    outputs=[]
                    coef_out=torch.zeros(coefs_lab.shape,device=device,dtype=torch.complex64)
                    coef_out_train=torch.zeros(coefs_lab_train.shape,device=device,dtype=torch.complex64)
                    
                    # Evaluate model for each frequency
                    for j in range(len(nu)):
                        coef_out_inter = model(x_data.unsqueeze(0),nu[j].unsqueeze(0))
                        coef_out[:,:,j]=coef_out_inter[0,:,:,0] 
                        coef_out_inter_train = model(x_data_train.unsqueeze(0),nu[j].unsqueeze(0))
                        coef_out_train[:,:,j]=coef_out_inter_train[0,:,:,0]
                    
                    # Convert predicted coefficients back to spatial grid
                    grids_pred=Calculate_inverse_grid(coef_out,Y_basis=Y_basis)
                    grids_pred_train=Calculate_inverse_grid(coef_out_train,Y_basis=Y_basis)
                    
                    # Initialize arrays for HRTF comparison
                    HRTF_GT=np.zeros((len(index_inci),256))
                    HRTF_pred=np.zeros((len(index_inci),256))
                    HRTF_GT_train=np.zeros((len(index_inci),256))
                    HRTF_pred_train=np.zeros((len(index_inci),256))   
                    inci_plot=[]
                    inci_plot_train=[]           
                    
                    # Extract HRTF for selected incidence directions
                    for i in range(len(index_inci)):
                        HRTF_GT[i,0:128]=grids_gt[index_inci[i],0,:]
                        HRTF_GT[i,128:256]=grids_gt[index_inci[i],1,:]
                        HRTF_GT_train[i,0:128]=grids_gt_train[index_inci[i],0,:]
                        HRTF_GT_train[i,128:256]=grids_gt_train[index_inci[i],1,:]
                        HRTF_pred[i,0:128]=grids_pred[index_inci[i],0,:]
                        HRTF_pred[i,128:256]=grids_pred[index_inci[i],1,:]
                        HRTF_pred_train[i,0:128]=grids_pred_train[index_inci[i],0,:]
                        HRTF_pred_train[i,128:256]=grids_pred_train[index_inci[i],1,:]
                        inci_plot.append(incidence_data[index_inci[i],0:2].detach().numpy())
                        inci_plot_train.append(incidence_data[index_inci[i],0:2].detach().numpy())
                    
                    inci_plot=np.array(inci_plot)
                    inci_plot_train=np.array(inci_plot_train)
                    
                    # Generate and log HRTF evolution plots
                    fig=gen_image_hrtf_evolve(HRTF_GT,HRTF_pred,inci_plot,epoch+1,model_name,input_type)
                    fig_train=gen_image_hrtf_evolve(HRTF_GT_train,HRTF_pred_train,inci_plot_train,epoch+1,model_name,input_type)
                    run.log({"HRTF_evolution_val_data": wandb.Image(fig)})
                    run.log({"HRTF_evolution_train_data": wandb.Image(fig_train)})
        
                    # === Global evaluation on entire validation set ===
                    N=dataset_val.__len__()
                    pred_HRTF=np.zeros((N,793,256))
                    GT_HRTF=np.zeros((N,793,256))
                    shape=torch.tensor(x_data.shape)
                    out_dim=torch.cat((torch.tensor(N).unsqueeze(0),shape),dim=0)
                    X_data=torch.zeros(list(out_dim))
                    
                    # Collect validation dataset
                    for j in range(N):
                        x_data,HRTF_gt=dataset_val.__getitem__(j,True)
                        inter=HRTF_gt.detach().numpy()
                        GT_HRTF[j,:,0:128]=inter[:,0,:]
                        GT_HRTF[j,:,128:256]=inter[:,1,:]
                        X_data[j,:]=x_data
                    
                    X_data=X_data.to(device)         
                    
                    # Evaluate model for all frequencies
                    for k in range(128):
                        freq= torch.ones((X_data.shape[0],1),device=device)*k
                        freq=freq.type(torch.long)
                        out=model(X_data,freq,out_HRTF=False)
                        # Convert SHT coefficients to spatial HRTF grid
                        inter2=Calculate_inverse_grid(out,Y_basis=Y_basis)
                        pred_HRTF[:,:,k]=inter2[:,:,0]
                        pred_HRTF[:,:,k+128]=inter2[:,:,1]
                    diff=(GT_HRTF-pred_HRTF)**2
            
            # === Generate global evaluation metrics and visualizations ===
            # Calculate global Log Spectral Distance
            global_LSD=np.sqrt(np.mean(diff))
            run.log({"Global_validation_LSD":global_LSD})        
            
            # Calculate directional statistics for left ear
            mean_direction_subject_left=np.sqrt(np.mean(diff[:,:,0:128],axis=(0,1)))
            std_direction_subject_left=np.sqrt(np.std(diff[:,:,0:128],axis=(0,1)))
            
            # Calculate directional statistics for right ear
            mean_direction_subject_right=np.sqrt(np.mean(diff[:,:,128:256],axis=(0,1)))
            std_direction_subject_right=np.sqrt(np.std(diff[:,:,128:256],axis=(0,1)))
            
            # Calculate directional statistics for both ears combined
            mean_direction_subject=np.sqrt(np.mean((diff[:,:,0:128]+diff[:,:,128:256])/2,axis=(0,1)))
            std_direction_subject=np.sqrt(np.std((diff[:,:,0:128]+diff[:,:,128:256])/2,axis=(0,1)))
            
            # Calculate frequency statistics for left ear
            mean_subject_frequency_left=np.sqrt(np.mean(diff[:,:,0:128],axis=(0,2)))
            std_subject_frequency_left=np.sqrt(np.std(diff[:,:,0:128],axis=(0,2)))
            
            # Calculate frequency statistics for right ear
            mean_subject_frequency_right=np.sqrt(np.mean(diff[:,:,128:256],axis=(0,2)))
            std_subject_frequency_right=np.sqrt(np.std(diff[:,:,128:256],axis=(0,2)))
            
            # Calculate frequency statistics for both ears combined
            mean_subject_frequency=np.sqrt(np.mean((diff[:,:,0:128]+diff[:,:,128:256])/2,axis=(0,2)))
            std_subject_frequency=np.sqrt(np.std((diff[:,:,0:128]+diff[:,:,128:256])/2,axis=(0,2)))
            
            # Generate global evaluation plots (combined ears, both ears separately, and per frequency)
            fig2=gen_image_global_evaluation_perf(std_direction_subject,std_subject_frequency,mean_direction_subject,mean_subject_frequency,input_type,epoch,model_name)
            fig3=gen_image_global_evaluation_perf(std_direction_subject_left,std_subject_frequency_left,mean_direction_subject_left,mean_subject_frequency_left,input_type,epoch,model_name+"_left_pred")
            fig4=gen_image_global_evaluation_perf(std_direction_subject_right,std_subject_frequency_right,mean_direction_subject_right,mean_subject_frequency_right,input_type,epoch,model_name+"_right_pred")
            
            # Log plots to WandB
            run.log({"Global evaluation left and right": wandb.Image(fig2)})
            run.log({"Global evaluation left": wandb.Image(fig3)})
            run.log({"Global evaluation right": wandb.Image(fig4)})
            
            # Close all matplotlib figures to free memory
            plt.close('all')
