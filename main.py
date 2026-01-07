
import argparse
from train import train
from optimization import evaluate

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    #------------------------------ 
    # General
    #------------------------------ 
    
    parser.add_argument('--exp_name', type=str, default='Anthro_to_HRTF', help='name of the experiment, can be any string')
    parser.add_argument('--train', type=str,default="True", help='whether to launch a training session or not')
    parser.add_argument('--evaluate', type=str,default="False", help='wheter to launch an hyperparameter optimization session or not')
    parser.add_argument('--Eval_on_data_type', type=str, default="False", help='wheter to "optunate" also on data type ("1d","2d","both","left)...')
    parser.add_argument('--n_trial', type=int, default=20, help='Number of trials in the hyperparameter optimization')
    #------------------------------ 
    # Network
    #------------------------------    
    parser.add_argument('--disto_photo', type=str, default='True', help='whether to use the distorted (224*224) ear photo or the non-distorded')
    parser.add_argument('--total_epochs', type=int, default=10,help='number of epochs to train the model or to run during an optimization trial')
    parser.add_argument('--batch_size', type=int, default=3, help='batch size of inference')
    parser.add_argument('--valid_batch_size', type=int, default=3, help='batch size of validation')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for the optimizer')
    parser.add_argument('--k_folds', type=int, default=5, help='number of folds for cross-validation')
    parser.add_argument('--test_fold', type=int, default=5, help='k for test')
    parser.add_argument('--model', type=str, default='Manlin-Zhao', help='model name among Le-roux, Woo-lee, Manlin-Zhao')
    parser.add_argument('--L_order_SHT', type=int, default=50, help='Maximum order of SHT if needed (value in 30 50 70 80 ) Only for Manlin-Zhao models')
    parser.add_argument('--N_Filters_le_roux', type=int, default=10, help='number of filters for the Le-roux model')
    parser.add_argument('--out_HRTF', type=str, default="True", help='to output the HRTF instead of other type of output (e.g. Impulse Response, SHT coefficients so only matters for Manlin-Zhao and Woo-lee models)')
    parser.add_argument("--hidden_sizes",type=int,nargs="+",default=[256, 256, 128,64,32],help="list of hidden layer sizes for the model (only for Le-roux models)")
    parser.add_argument('--dropout', type=float,nargs="+" ,default=[0.2,0.2,0.2], help='dropout rate for the model (only for Woo-lee models)')
    #------------------------------ 
    # Data
    #------------------------------ 
    parser.add_argument('--input_type', type=str, default='2d', help='input type: 1d, 2d, 3d being respectively ear distances, ear photos, ear 3d ear meshs')
    parser.add_argument('--ear_input', type=str, default='both', help='wich ear rep as input: left, right, both')
    parser.add_argument('--Model_dir', type=str, default='/Your_path_to/Model_weights/', help='Path to save or load the model weights')
    parser.add_argument('--ears_dataset_path', type=str, default='/Your_path_to/sonicom_ears_dataset/', help='path to the ears representation dataset')
    parser.add_argument('--HRTF_dataset_path', type=str, default='/your_path_to/sonicom_hrtf_dataset/', help='path to the HRTF dataset')
    parser.add_argument('--early_stop', type=str, default="True", help='whether to use early stopping')
    parser.add_argument('--percent_of_data', type=int, default=80, help='percentage of data to use for training')

    args = parser.parse_args()


    if args.train=="True":
        train(args)
    elif args.evaluate=="True":
        evaluate(args)
  