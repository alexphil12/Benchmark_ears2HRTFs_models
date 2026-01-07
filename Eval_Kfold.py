from train import train
import argparse
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, Subset
import torch
from Datasets import HRTF_mesh_dataset
from utils import init_weights
from torch import nn, optim
from utils import hrir_to_hrtf,gen_image_hrtf_evolve,Calculate_inverse_grid,EarlyStopping,gen_image_global_evaluation_perf,MSE_root,init_weights,training_loop
import os
import wandb
from pathlib import Path
import matplotlib.pyplot as plt


def create_kfold_splits(dataset: Dataset, n_splits: int = 5):
    """
    Crée des splits entraînement/validation pour la validation croisée en K-Folds 
    à partir d'un dataset PyTorch.

    Args:
        dataset (torch.utils.data.Dataset): Le dataset PyTorch complet.
        n_splits (int): Le nombre de folds (K). Par défaut est 5.

    Yields:
        tuple: Un tuple contenant les datasets PyTorch pour l'entraînement et la validation 
               pour le fold courant (train_subset, val_subset).
    """

    indices = np.arange(len(dataset))

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=41)

    for fold, (train_indices, val_indices) in enumerate(kfold.split(indices)):
        print(f"--- Fold {fold + 1}/{n_splits} ---")
        
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)
        
        print(f"  Taille entraînement : {len(train_subset)}")
        print(f"  Taille validation : {len(val_subset)}")
        
        yield train_subset, val_subset


if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    #------------------------------ 
    # General
    #------------------------------ 
    
    parser.add_argument('--exp_name', type=str, default='Anthro_to_HRTF')
    parser.add_argument('--train', type=str,default="True", help='train the model')
    parser.add_argument('--evaluate', type=str,default="False", help='test the model')
    parser.add_argument('--Eval_on_data_type', type=str, default="False", help='wheter to "optunate" also on data type ("1d","2d","both","left)...')
    parser.add_argument('--n_trial', type=int, default=20, help='Number of trials for the evaluation')
    parser.add_argument('--k_folds', type=int, default=5, help='number of folds for cross-validation')
    #------------------------------ 
    # Network
    #------------------------------    
    parser.add_argument('--disto_photo', type=str, default='True', help='whether to use the distorted (224*224) ear photo or the non-distorded')
    parser.add_argument('--total_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=18, help='batch size of inference')
    parser.add_argument('--valid_batch_size', type=int, default=18, help='batxh size of validation')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--k_folds', type=int, default=5, help='number of folds for cross-validation')
    parser.add_argument('--test_fold', type=int, default=5, help='k for test')
    parser.add_argument('--model', type=str, default='Woo-lee', help='model name among Le-roux, Woo-lee, Manlin-Zhao, Manlin-Zhao-freq')
    parser.add_argument('--L_order_SHT', type=int, default=50, help='Maximum order of SHT if needed (value in 30 50 70 80 )')
    parser.add_argument('--N_Filters_le_roux', type=int, default=10, help='number of filters for the Le-roux model')
    parser.add_argument('--out_HRTF', type=str, default="True", help='to output the HRTF instead of other type of output (e.g. Impulse Response, SHT coefficients,)')
    parser.add_argument("--hidden_sizes",type=int,nargs="+",default=[256, 256, 128,64,32],help="list of hidden layer sizes for the model")
    parser.add_argument('--dropout', type=float,nargs="+" ,default=[0.2,0.2,0.2], help='dropout rate for the model')
    #------------------------------ 
    # Data
    #------------------------------ 
    parser.add_argument('--input_type', type=str, default='2d', help='input type: 1d, 2d, 3d')
    parser.add_argument('--ear_input', type=str, default='both', help='wich ear rep as input: left, right, both')
    parser.add_argument('--log_tensorboard_dir', type=str, default='/home/PhilipponA/Model_anthro_HRTF/Tensorboard/', help='')
    parser.add_argument('--Model_dir', type=str, default='/home/PhilipponA/Model_anthro_HRTF/Model_weights/', help='')
    parser.add_argument('--dataset_path', type=str, default='/home/PhilipponA/HRTF_sonicom_dataset/', help='path to the dataset')
    parser.add_argument('--early_stop', type=str, default="False", help='whether to use early stopping')
    parser.add_argument('--percent_of_data', type=int, default=80, help='percentage of data to use for training')

    args = parser.parse_args()

    hidden_sizes=args.hidden_sizes
    Name_exp=args.exp_name
    dropout=args.dropout
    Train_mode=args.train
    Test_mode=args.evaluate  
    num_epochs=args.total_epochs
    batch_size=args.batch_size
    lr=args.lr
    k_folds=args.k_folds
    test_fold=args.test_fold
    model_name=args.model
    early_stoping=args.early_stop
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
    if out_HRTF=="True":
        out_HRTF=True
    else:
        out_HRTF=False
    distorted_photo=args.disto_photo
    if distorted_photo=="True":
        distorted_photo=True
    else:
        distorted_photo=False
    bool_forward=False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Y_basis=torch.tensor(np.load("/home/PhilipponA/Script_model_HRTF_Sonicom/Spherical_based_save/spherical Base N="+str(L_order_SHT)+".npy"),dtype=torch.complex64)
    Y_basis=Y_basis.to(device)  
    if model_name=="Woo-lee":
        output_type="IR_dir"
    elif model_name=="Le-roux":
        output_type="HRTF_dir"
    elif model_name=="Manlin-Zhao":
        output_type="SHT_set"     
    else:
        raise ValueError("Invalid model name. Choose 'Woo-lee', 'Le-roux', or 'Manlin-Zhao'.")
    input_type=args.input_type
    if input_type=="1d":
        whole_input="ears_measurements"
    elif input_type=="2d":
        whole_input="ears_photo"
    elif input_type=="3d":
        whole_input="Ears_3D_voxels"

    whole_xp_name="/"+Name_exp+"_model_"+model_name


    K_FOLDS=args.k_folds
    full_dataset=HRTF_mesh_dataset(type_of_data=input_type,output_type=output_type,Train_data=True,L=L_order_SHT,mode=ear_input,distorted_photo=distorted_photo,percent_of_data=100,Y_basis=Y_basis,device=device,out_HRTF=out_HRTF)
    incidence_data=full_dataset.incidence_dir
    if model_name=="Woo-lee":
        from models.Woo_Lee_simple_model import Woo_lee_Model
        model = Woo_lee_Model(input_type=input_type,ear_input=ear_input,dropout_rates=dropout,distorted_photo=distorted_photo)
    elif model_name=="Le-roux":
        from models.Le_roux_model_ddsp import HRTFEstimator_Le_roux
        model = HRTFEstimator_Le_roux(hidden_sizes=hidden_sizes, num_filters=Number_of_filters, n_freqs=128,input_type=input_type,ear_input=ear_input,dropout_rates=dropout,distorted_photo=distorted_photo)
    elif model_name=="Manlin-Zhao":
        from models.Manlin_Zhao_model_sht import Manlin_Zhao_Model
        model = Manlin_Zhao_Model(input_type=input_type,order_of_sht=L_order_SHT,num_of_frequency=128,ear_input=ear_input,incidence=incidence_data,distorted_photo=distorted_photo,Y_basis=Y_basis)
    model.apply(init_weights)
    model = model.to(device)
    criterion = MSE_root()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    nu=np.linspace(0,44100/2,128)
    whole_xp_name="/"+"K_FOLDS_evaluation"+"_model_"+model_name
    path_model_weight = args.Model_dir+whole_xp_name+"_"+input_type
    if not os.path.exists(args.Model_dir+whole_xp_name+"_"+input_type):
        os.makedirs(args.Model_dir+whole_xp_name+"_"+input_type)

    file_path = Path('clef_wandb.txt')

    wan_key = file_path.read_text()

    wandb.login(key=wan_key)
    disto_label="distorted_photo" if distorted_photo else "no_distorted_photo"
    run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="alexandre-philippon-universite-de-mons",
    project="K_FOLDS_evaluation_Benchmark_models",
    config={"learning_rate": lr,"architecture":model_name ,"dataset": "Sonicom",
        "epochs": num_epochs,"batch_size": batch_size,"input_type": input_type,
        "output_type": output_type,"ear_input": ear_input,"N_filters_le_roux": Number_of_filters,"SHT_decomposition_order":L_order_SHT,"type_of_2D_input":disto_label}
)
    All_fold_train_losses=np.zeros((K_FOLDS,num_epochs))
    All_fold_eval_losses=np.zeros((K_FOLDS,num_epochs))
    best_losses_folds=np.zeros(K_FOLDS)
    for fold_idx, (train_set, val_set) in enumerate(create_kfold_splits(full_dataset, K_FOLDS)):
        print(f"Début de l'entraînement pour le Fold {fold_idx + 1}...")
        from torch.utils.data import DataLoader
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=valid_batch_size, shuffle=False)
        
        # ... (Definition of the model, optimizer, and training loop) ...
        train_losses, eval_losses, best_loss = training_loop(num_epochs, incidence_data, train_loader, val_loader, model, criterion, optimizer, device, output_type, out_HRTF, nu=nu, early_stoping=early_stoping, path_model_weight=path_model_weight, run=run, fold_idx=fold_idx,batch_size=batch_size,valid_batch_size=valid_batch_size)
        All_fold_train_losses[fold_idx, :] = np.array(train_losses)
        All_fold_eval_losses[fold_idx, :] = np.array(eval_losses)
        best_losses_folds[fold_idx] = best_loss
        model.apply(init_weights)
    
        # For the example, we will just display the size and move on to the next
        print(f"Fin de l'entraînement du Fold {fold_idx + 1}.\n")
    fig,ax =plt.subplots()
    ax.plot(np.arange(1,num_epochs+1),np.mean(All_fold_train_losses,axis=0),label="Train loss",color='blue')
    ax.plot(np.arange(1,num_epochs+1),np.mean(All_fold_eval_losses,axis=0),label="Eval loss",color='red')
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title(f"Mean Train and Eval Losses over {K_FOLDS} Folds for {model_name} model")
    ax.legend()
    run.log({"mean_best_losses_folds":np.mean(best_losses_folds),
             f"loss_curves_{model_name}_{input_type}":wandb.Image(fig)})
    run.finish()




