# Benchmark of AI model estimating HRTFs from ears representation #
![Screenshot of the scheme resuming the idea of the benchmark](https://github.com/alexphil12/Script_model_HRTF_Sonicom/blob/master/Capture%20d'%C3%A9cran%202026-01-05%20180057.png)

This repository containt the code of the following paper:
> Benchmark for estimating HRTFs An evaluation benchmark of artificial intelligence models for estimating head-related transfer functions (HRTFs) from ear shape representations
## Setup ##
You can create a Conda environment
```
conda create -n HRTF_bench_env python=3.10
conda activate HRTF_bench_env
```
You need to install pytorch and Cuda toolkit and also the dependencies using
```
git clone https://github.com/alexphil12/Script_model_HRTF_Sonicom.git
cd Script_model_HRTF_Sonicom
pip install -r requirements.txt -e .
```
## Data ##
The Dataset originaly used for those experiments comes from the SONICOM HRTF dataset: [link to original data](https://www.sonicom.eu/tools-and-resources/hrtf-dataset/)

The preprocessed ears data (input data for the models) used in our experiments can be found at :[link to preprocessed data](https://www.dropbox.com/scl/fi/hjaryg21p4ijik29ggumk/download_2026-01-05_17-40-42.zip?rlkey=53rqkbe2dq5sic3xk4r69yasj&st=7dgzsyix&dl=0) 

You need to dowload the data from the SONICOM HRTF dataset from subject 2 to 100 (only the HRTF folders is necessary), to put those folder into a folder called "sonicom_hrtf_dataset" and to modify accordingbly the variable "hrtf_dataset_path" in the main.py argument parser.

Note that the 2D datas (ear photo) are not provided as their are under restricted use. If you wish to acess them, please contact Mr Turvey at o.turvey22@imperial.ac.uk  .

Once downloaded, the data needs to be unziped and the Folder "Mesh_3D_ears" and "Measure_1D_ears" need to be put in a folder Named HRTF_sonicom_dataset, you will have then to modify your "ears_dataset_path" variable default in the main.py argument parser.

## Run a training ##

Once the data correctly set up, you can run a training by executing the main.py files with the 'train' parameter set with "True", You can modify parameters such as "model", "input_type", "batch_size" etc. All the detail on how to use those are explained on the main.py file. See some example of valid training command bellow.

```
python main.py --model 'Le-roux' --input_type '1d' --ear_input 'right' --exp_name "measure_to_HRTF" --early_stop "True" --batch_size 15 --N_Filters_le_roux 20 --train "True" --evaluate "False" --total_epochs 100 
python main.py --model 'Manlin-Zhao' --input_type '2d' --ear_input 'both' --disto_photo "True"  --exp_name "photos_to_HRTFs" --early_stop "True" --batch_size 15 --L_order_SHT 17 --train "True"  --total_epochs 200
python main.py --model 'Woo-lee' --lr 2.4e-4 --input_type '3d' --ear_input 'left' --exp_name "voxel_to_HRTF" --early_stop "False" --batch_size 12 --valid_batch_size 3 --train "True" --total_epochs 60  --out_HRTF "True" --dropout 0.15 0.15 0.1
````
note: the monitoring of the losses and other data is implemented with WandB, you will then need to modify the train.py file to loggin your wandb and load the data into your project.

## Run an hyper-parameter optimization ##

Same process as for running a training this time with the "evaluate" parameter set as "True". Most of the parameter used for training will be ignored (detaille about those one more time in the main.py).
see bellow some example of valid command.
```
python main.py --model 'Manlin-Zhao' --input_type '1d' --ear_input 'right' --disto_photo "True"  --exp_name "distance_to_HRTF_right_in" --early_stop "True" --batch_size 15 --L_order_SHT 10 --train "False" --evaluate "True" --total_epochs 30 --Eval_on_data_type "False" --n_trial 40
python main.py --model 'Woo-lee' --input_type '2d' --ear_input 'both' --exp_name "photo_to_HRTF" --early_stop "True" --batch_size 15 --L_order_SHT 17 --train "False" --evaluate "True" --total_epochs 30 --out_HRTF "False" --Eval_on_data_type "False" --disto_photo "False" --n_trial 9
python main.py --model 'Le-roux' --input_type '2d' --ear_input 'left' --disto_photo "False" --exp_name "photo_to_HRTF" --early_stop "True" --batch_size 15 --N_Filters_le_roux 20 --train "False" --evaluate "True" --total_epochs 30 --Eval_on_data_type "False" --n_trial 18
```
You can find the detail about the availlable hyper-parameter in the optimization.py file. (optimization is made using optuna framework). Same remark concerning the WandB monitoring, you need to adapt it with your credentials.

## Run a K-fold evaluation ##
To process this K-fold evaluation, you will have to run the Eval_Kfold.py file this time, and to precisse your parameters, note that by default, the evaluation is conducted on 5 folds. Some examples of valid commands.
```
python Eval_Kfold.py --model 'Le-roux' --lr 2e-5 --input_type '1d' --ear_input 'left' --exp_name "measure_to_HRTF" --early_stop "False" --batch_size 15 --N_Filters_le_roux 25 --train "True" --evaluate "False" --total_epochs 30   --hidden_sizes 192 128 192 160 80
python Eval_Kfold.py --lr 1e-3 --model 'Manlin-Zhao' --input_type '2d' --disto_photo "True" --out_HRTF "True" --L_order_SHT 17 --ear_input 'both' --exp_name "image_to_HRTF_left_in_true_disto" --early_stop "False" --batch_size 15  --train "True" --evaluate "False" --total_epochs 30
python Eval_Kfold.py --model 'Woo-lee' --lr 2.4e-4 --input_type '3d' --ear_input 'right' --exp_name "voxel_to_HRTF_true" --early_stop "False" --batch_size 15 --train "True" --evaluate "False" --total_epochs 30  --out_HRTF "True" --dropout 0.15 0.15 0.1
```
## Prediction example ##
![Screenshot of the prediction of HRTF by the models](https://github.com/alexphil12/Script_model_HRTF_Sonicom/blob/master/visualization/prediction_example.png)
