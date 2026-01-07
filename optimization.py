import torch.optim as optim
import torch
import optuna
import wandb
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Datasets import HRTF_mesh_dataset
from tqdm import tqdm
import numpy as np
from functools import partial
from optuna.trial import TrialState
from utils import init_weights

def objective(trial, args):
    """
    Optuna objective function for hyperparameter optimization of HRTF models.
    
    Args:
        trial: Optuna trial object for suggesting hyperparameters
        args: Configuration arguments
    
    Returns:
        float: Global minimum evaluation loss
    """
    # Extract configuration parameters from args
    Name_exp = args.exp_name
    Train_mode = args.train
    Test_mode = args.evaluate
    ears_dataset_path = args.ears_dataset_path
    HRTF_dataset_path = args.HRTF_dataset_path  
    num_epochs = args.total_epochs
    disto_photo = args.disto_photo
    batch_size = args.batch_size
    evaluate_mode = args.Eval_on_data_type
    lr = args.lr
    percent_of_data = args.percent_of_data
    k_folds = args.k_folds
    test_fold = args.test_fold
    out_HRTF = args.out_HRTF
    
    # Convert string boolean flags to actual booleans
    if disto_photo == "True":
        disto_photo = True
    else:
        disto_photo = False
    if out_HRTF == "True":
        out_HRTF = True
    else:
        out_HRTF = False
    
    # Model and training configuration
    model_name = args.model
    early_stoping = args.early_stop
    valid_batch_size = args.valid_batch_size
    ear_input = args.ear_input  
    L_order_SHT = args.L_order_SHT
    Number_of_filters = args.N_Filters_le_roux
    
    # Map model name to output type
    if model_name == "Woo-lee":
        output_type = "IR_dir"
    elif model_name == "Le-roux":
        output_type = "HRTF_dir"
    elif model_name == "Manlin-Zhao":
        output_type = "SHT_set"
    else:
        raise ValueError("Invalid model name. Choose 'Woo-lee', 'Le-roux', or 'Manlin-Zhao'.")
    
    input_type = args.input_type
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert string boolean to actual boolean
    if evaluate_mode == "True":
        evaluate_mode = True
    else:
        evaluate_mode = False

    # ===== Le-roux Model Configuration =====
    if args.model == 'Le-roux':
        # Load spherical basis functions
        Y_basis = torch.tensor(np.load("/home/PhilipponA/Script_model_HRTF_Sonicom/Spherical_based_save/spherical Base N=17.npy"), dtype=torch.complex64)
        Y_basis = Y_basis.to(device)  
        from models.Le_roux_model_ddsp import HRTFEstimator_Le_roux
        
        # Suggest hyperparameters for Le-roux model
        dropout_rate_1 = trial.suggest_float("dropout_rate", 0, 0.5, step=0.05)
        dropout_rate_2 = trial.suggest_float("dropout_rate_2", 0, 0.5, step=0.05)
        hidden_size_1 = trial.suggest_int("hidden_size_1", 64, 512, step=64)
        hidden_size_2 = trial.suggest_int("hidden_size_2", 64, 512, step=64)
        hidden_size_3 = trial.suggest_int("hidden_size_3", 32, 256, step=32)
        hidden_size_4 = trial.suggest_int("hidden_size_4", 32, 256, step=32)
        hidden_size_5 = trial.suggest_int("hidden_size_5", 16, 128, step=16)
        hidden_sizes_try = [hidden_size_1, hidden_size_2, hidden_size_3, hidden_size_4, hidden_size_5]
        dropout_rates_try = [dropout_rate_1, dropout_rate_2]   
        N_filters_try = trial.suggest_int("N_filters", 4, 32, step=1)
        
        # If evaluating, suggest input/output configuration; otherwise use fixed config
        if evaluate_mode:
            ear_input_try = trial.suggest_categorical("ear_input", ["both", "left", "right"])
            input_type_try = trial.suggest_categorical("input_type", ["1d", "2d", "3d"])
            disto_photo_try = trial.suggest_categorical("distorted_photo", [True, False])
        else:
            ear_input_try = ear_input
            input_type_try = input_type
            disto_photo_try = disto_photo

        # Create train and test datasets
        dataset_train = HRTF_mesh_dataset(ears_dataset_path=ears_dataset_path, HRTF_dataset_path=HRTF_dataset_path, type_of_data=input_type, output_type=output_type, Train_data=True, L=L_order_SHT, mode=ear_input, distorted_photo=disto_photo_try, percent_of_data=percent_of_data, Y_basis=Y_basis, device=device)
        dataset_test = HRTF_mesh_dataset(ears_dataset_path=ears_dataset_path, HRTF_dataset_path=HRTF_dataset_path, type_of_data=input_type, output_type=output_type, Train_data=False, L=L_order_SHT, mode=ear_input, distorted_photo=disto_photo_try, percent_of_data=percent_of_data, Y_basis=Y_basis, device=device)
    
        # Create datasets with HRTF output
        dataset_train_out_HRTF = HRTF_mesh_dataset(ears_dataset_path=ears_dataset_path, HRTF_dataset_path=HRTF_dataset_path, type_of_data=input_type, output_type=output_type, Train_data=True, L=L_order_SHT, mode=ear_input, out_HRTF=out_HRTF, distorted_photo=disto_photo_try, percent_of_data=percent_of_data, Y_basis=Y_basis, device=device)
        dataset_test_out_HRTF = HRTF_mesh_dataset(ears_dataset_path=ears_dataset_path, HRTF_dataset_path=HRTF_dataset_path, type_of_data=input_type, output_type=output_type, Train_data=False, L=L_order_SHT, mode=ear_input, out_HRTF=out_HRTF, distorted_photo=disto_photo_try, percent_of_data=percent_of_data, Y_basis=Y_basis, device=device)
        
        # Select appropriate dataset based on output type
        if out_HRTF:
            DataLoader_train = DataLoader(dataset_train_out_HRTF, batch_size=batch_size, shuffle=True)
            DataLoader_test = DataLoader(dataset_test_out_HRTF, batch_size=valid_batch_size, shuffle=False)
        else:
            DataLoader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
            DataLoader_test = DataLoader(dataset_test, batch_size=valid_batch_size, shuffle=False)
        incidence_data = dataset_train.incidence_dir

        # Initialize Le-roux model
        model = HRTFEstimator_Le_roux(
            hidden_sizes=hidden_sizes_try,
            num_filters=N_filters_try,
            n_freqs=128,
            input_type=input_type_try,
            ear_input=ear_input_try,
            dropout_rates=dropout_rates_try,
            distorted_photo=disto_photo,
        )
        
    # ===== Woo-lee Model Configuration =====
    elif args.model == 'Woo-lee':
        # Load spherical basis functions
        Y_basis = torch.tensor(np.load("/home/PhilipponA/Script_model_HRTF_Sonicom/Spherical_based_save/spherical Base N=17.npy"), dtype=torch.complex64)
        Y_basis = Y_basis.to(device)
        from models.Woo_Lee_simple_model import Woo_lee_Model
        
        # Suggest dropout rates for Woo-lee model
        dropout_rate_1 = trial.suggest_float("dropout_rate_1", 0, 0.2, step=0.05)
        dropout_rate_2 = trial.suggest_float("dropout_rate_2", 0, 0.2, step=0.05)
        dropout_rate_3 = trial.suggest_float("dropout_rate_3", 0, 0.2, step=0.05)
        dropout_rates = [dropout_rate_1, dropout_rate_2, dropout_rate_3]
        
        # If evaluating, suggest configuration; otherwise use fixed config
        if evaluate_mode:
            ear_input_try = trial.suggest_categorical("ear_input", ["both", "left", "right"])
            input_type_try = trial.suggest_categorical("input_type", ["1d", "2d", "3d"])
            disto_photo_try = trial.suggest_categorical("distorted_photo", [True, False])
        else:
            ear_input_try = ear_input
            input_type_try = input_type
            disto_photo_try = disto_photo

        # Create datasets
        dataset_train = HRTF_mesh_dataset(ears_dataset_path=ears_dataset_path, HRTF_dataset_path=HRTF_dataset_path, type_of_data=input_type, output_type=output_type, Train_data=True, L=L_order_SHT, mode=ear_input, distorted_photo=disto_photo_try, percent_of_data=percent_of_data, Y_basis=Y_basis, device=device)
        dataset_test = HRTF_mesh_dataset(ears_dataset_path=ears_dataset_path, HRTF_dataset_path=HRTF_dataset_path, type_of_data=input_type, output_type=output_type, Train_data=False, L=L_order_SHT, mode=ear_input, distorted_photo=disto_photo_try, percent_of_data=percent_of_data, Y_basis=Y_basis, device=device)
    
        dataset_train_out_HRTF = HRTF_mesh_dataset(ears_dataset_path=ears_dataset_path, HRTF_dataset_path=HRTF_dataset_path, type_of_data=input_type, output_type=output_type, Train_data=True, L=L_order_SHT, mode=ear_input, out_HRTF=out_HRTF, distorted_photo=disto_photo_try, percent_of_data=percent_of_data, Y_basis=Y_basis, device=device)
        dataset_test_out_HRTF = HRTF_mesh_dataset(ears_dataset_path=ears_dataset_path, HRTF_dataset_path=HRTF_dataset_path, type_of_data=input_type, output_type=output_type, Train_data=False, L=L_order_SHT, mode=ear_input, out_HRTF=out_HRTF, distorted_photo=disto_photo_try, percent_of_data=percent_of_data, Y_basis=Y_basis, device=device)
        incidence_data = dataset_train.incidence_dir
        
        # Select appropriate dataset
        if out_HRTF:
            DataLoader_train = DataLoader(dataset_train_out_HRTF, batch_size=batch_size, shuffle=True)
            DataLoader_test = DataLoader(dataset_test_out_HRTF, batch_size=valid_batch_size, shuffle=False)
        else:
            DataLoader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
            DataLoader_test = DataLoader(dataset_test, batch_size=valid_batch_size, shuffle=False)
        
        # Initialize Woo-lee model
        model = Woo_lee_Model(input_type=input_type_try, ear_input=ear_input_try, dropout_rates=dropout_rates, distorted_photo=disto_photo)
        
    # ===== Manlin-Zhao Model Configuration =====
    elif args.model == 'Manlin-Zhao':
        from models.Manlin_Zhao_model_sht import Manlin_Zhao_Model
        
        # Suggest SHT order parameter
        L_order_SHT_try = trial.suggest_categorical("L_order_SHT", [17, 30, 50, 70, 80])
        
        # Load corresponding spherical basis
        Y_basis = torch.tensor(np.load("/home/PhilipponA/Script_model_HRTF_Sonicom/Spherical_based_save/spherical Base N=" + str(L_order_SHT_try) + ".npy"), dtype=torch.complex64)
        Y_basis = Y_basis.to(device)  
        
        # If evaluating, suggest configuration; otherwise use fixed config
        if evaluate_mode:
            ear_input_try = trial.suggest_categorical("ear_input", ["both", "left", "right"])
            input_type_try = trial.suggest_categorical("input_type", ["1d", "2d", "3d"])
            disto_photo_try = trial.suggest_categorical("distorted_photo", [True, False])
        else:
            ear_input_try = ear_input
            input_type_try = input_type
            disto_photo_try = disto_photo
        
        # Create datasets
        dataset_train = HRTF_mesh_dataset(ears_dataset_path=ears_dataset_path, HRTF_dataset_path=HRTF_dataset_path, type_of_data=input_type, output_type=output_type, Train_data=True, L=L_order_SHT, mode=ear_input, distorted_photo=disto_photo_try, percent_of_data=percent_of_data, Y_basis=Y_basis, device=device)
        dataset_test = HRTF_mesh_dataset(ears_dataset_path=ears_dataset_path, HRTF_dataset_path=HRTF_dataset_path, type_of_data=input_type, output_type=output_type, Train_data=False, L=L_order_SHT, mode=ear_input, distorted_photo=disto_photo_try, percent_of_data=percent_of_data, Y_basis=Y_basis, device=device)
    
        dataset_train_out_HRTF = HRTF_mesh_dataset(ears_dataset_path=ears_dataset_path, HRTF_dataset_path=HRTF_dataset_path, type_of_data=input_type, output_type=output_type, Train_data=True, L=L_order_SHT, mode=ear_input, out_HRTF=out_HRTF, distorted_photo=disto_photo_try, percent_of_data=percent_of_data, Y_basis=Y_basis, device=device)
        dataset_test_out_HRTF = HRTF_mesh_dataset(ears_dataset_path=ears_dataset_path, HRTF_dataset_path=HRTF_dataset_path, type_of_data=input_type, output_type=output_type, Train_data=False, L=L_order_SHT, mode=ear_input, out_HRTF=out_HRTF, distorted_photo=disto_photo_try, percent_of_data=percent_of_data, Y_basis=Y_basis, device=device)
        incidence_data = dataset_train.incidence_dir
        
        # Select appropriate dataset
        if out_HRTF:
            DataLoader_train = DataLoader(dataset_train_out_HRTF, batch_size=batch_size, shuffle=True)
            DataLoader_test = DataLoader(dataset_test_out_HRTF, batch_size=valid_batch_size, shuffle=False)
        else:
            DataLoader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
            DataLoader_test = DataLoader(dataset_test, batch_size=valid_batch_size, shuffle=False)
        
        # Initialize Manlin-Zhao model
        model = Manlin_Zhao_Model(
            input_type=input_type_try,
            order_of_sht=L_order_SHT_try,
            Y_basis=Y_basis,
            num_of_frequency=128,
            incidence=incidence_data,
            ear_input=ear_input_try,
            distorted_photo=disto_photo_try
        )
    else:
        raise ValueError("Invalid model name. Choose 'Le-roux', 'Woo-lee', or 'Manlin-Zhao'.")
    
    # ===== Optimizer and Training Setup =====
    # Suggest optimizer (currently only Adam)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam"])
    
    # Suggest learning rate
    lr = trial.suggest_float("lr", 1e-7, 1e-3, log=True)
    
    # Initialize optimizer
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Create Weights & Biases config dictionary
    config = dict(trial.params)
    config["trial.number"] = trial.number
    config["architecture"] = model_name
    config["dataset"] = "Sonicom"
    config["distorted_photo"] = "distorted_photo" if disto_photo_try else "no_distorted_photo"
    config["epochs"] = num_epochs
    config["batch_size"] = batch_size
    config["input_type"] = input_type if evaluate_mode == False else input_type_try
    config["output_type"] = output_type
    config["ear_input"] = ear_input if evaluate_mode == False else ear_input_try
    config["L_order_SHT"] = L_order_SHT_try if model_name == 'Manlin-Zhao' else None

    # ===== Weights & Biases Setup =====
    from pathlib import Path
    
    # Load API key from file
    file_path = Path('clef_wandb.txt')
    wan_key = file_path.read_text()
    
    # Login and initialize wandb run
    wandb.login(key=wan_key)
    run = wandb.init(
        entity="alexandre-philippon-universite-de-mons",
        project="Sonicom-benchmark-hyperparameters-opti " + model_name,
        config=config,
        reinit="return_previous",
    )
    
    # ===== Model Training =====
    # Initialize model weights
    model.apply(init_weights)
    model = model.to(device)
    
    # Frequency bins for SHT models
    nu = np.linspace(0, 44100/2, 128)
    
    # Track minimum evaluation loss
    global_min_eval = float('inf')
    
    # Training loop
    for epoch in range(num_epochs):
        index_inci = list(range(len(incidence_data)))
        model.train()
        running_loss = 0.0
        loop = tqdm(DataLoader_train, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        
        # Training batches
        for inputs, labels in loop:
            X_data, y_labels = inputs.to(device), labels.to(device)
            
            # Random permutations for data augmentation
            perm = np.random.permutation(index_inci)
            perm_nu = np.random.permutation(range(len(nu)))
            
            # Forward pass depends on output type
            if "dir" in output_type:
                # For directional outputs (IR_dir, HRTF_dir)
                for i in range(len(incidence_data)):
                    inci_gpu = incidence_data.to(device)
                    inci_gpu = inci_gpu[perm[i], :]
                    inci_gpu = inci_gpu.repeat((X_data.shape[0], 1)) / 360
                    
                    # Forward pass
                    outputs = model(X_data, inci_gpu, out_HRTF=out_HRTF)
                    loss = criterion(outputs, y_labels[:, perm[i], :, :].reshape_as(outputs))
                    
                    # Backpropagation
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1000)
                    optimizer.step()
                    running_loss += loss.item()
                running_loss /= len(inci_gpu)
                
            elif output_type == "SHT_set":
                # For spherical harmonics coefficient outputs
                for i in range(len(nu)):
                    freq = torch.ones((X_data.shape[0], 1), device=device) * perm_nu[i]
                    freq = freq.type(torch.long)   
                    
                    # Forward pass
                    outputs = model(X_data, freq, out_HRTF=out_HRTF)
                    loss = criterion(torch.view_as_real(outputs), torch.view_as_real(y_labels[:, :, :, :, perm_nu[i]].reshape_as(outputs)))
                    
                    # Backpropagation
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1000)
                    optimizer.step()
                    running_loss += loss.item()
                running_loss /= len(nu)                 
            
            loop.set_postfix(loss=running_loss)
        
        # Average epoch loss
        epoch_loss = running_loss / len(DataLoader_train)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        
        # ===== Evaluation Phase =====
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            torch.cuda.empty_cache()
            loop_eval = tqdm(DataLoader_test, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
            running_loss_eval = 0.0
            
            # Test batches
            for inputs, labels in loop_eval:
                X_data_ev, y_labels_ev = inputs.to(device), labels.to(device)
                perm = np.random.permutation(index_inci)
                
                # Evaluation depends on output type
                if "dir" in output_type:
                    for i in range(len(incidence_data)):
                        inci_gpu = incidence_data.to(device)
                        inci_gpu = inci_gpu[perm[i], :]
                        inci_gpu = inci_gpu.repeat((X_data_ev.shape[0], 1)) / 360
                        
                        # Forward pass
                        outputs = model(X_data_ev, inci_gpu, out_HRTF=out_HRTF)
                        loss = criterion(outputs, y_labels_ev[:, perm[i], :, :].reshape_as(outputs))
                        running_loss_eval += loss.item()
                    running_loss_eval /= len(inci_gpu)
                    
                elif output_type == "SHT_set":
                    for i in range(len(nu)):
                        freq = torch.ones((X_data_ev.shape[0], 1), device=device) * perm_nu[i]
                        freq = freq.type(torch.long)   
                        
                        # Forward pass
                        outputs = model(X_data_ev, freq, out_HRTF=out_HRTF)
                        loss = criterion(torch.view_as_real(outputs), torch.view_as_real(y_labels_ev[:, :, :, :, perm_nu[i]].reshape_as(outputs)))
                        running_loss_eval += loss.item()
                    running_loss_eval /= len(nu)
                
                loop_eval.set_postfix(loss=running_loss_eval)
            
            # Average test loss
            test_loss = running_loss_eval / (len(DataLoader_test))
            
            # Log metrics to wandb
            run.log({"test-loss": test_loss, "train-loss": epoch_loss})
            
            # Track best model
            if test_loss < global_min_eval:
                global_min_eval = test_loss
                print(f"Model saved at epoch {epoch+1} with test loss {test_loss:.4f}")
            
            print(f"Epoch {epoch+1}/{num_epochs}, test Loss: {test_loss:.4f}") 
            
            # Check if trial should be pruned
            if trial.should_prune():
                wandb.run.summary["state"] = "pruned"
                wandb.finish(quiet=True)
                raise optuna.exceptions.TrialPruned()
    
    # ===== Cleanup =====
    wandb.run.summary["state"] = "completed"
    wandb.finish(quiet=True)
    
    # Return best evaluation loss
    return global_min_eval

def evaluate(args):
    pruner_med =optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=10)
    study = optuna.create_study(direction="minimize",study_name=args.exp_name,sampler=optuna.samplers.TPESampler(),pruner=pruner_med)
    n_trials=args.n_trial
    study.optimize(lambda trial: objective(trial,args), n_trials=n_trials, timeout=None,catch=(ValueError, RuntimeError, optuna.exceptions.TrialPruned),)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
   
