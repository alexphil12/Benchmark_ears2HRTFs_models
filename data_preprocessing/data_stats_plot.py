import numpy as np

import matplotlib.pyplot as plt

stats_1D_L=np.load('/home/PhilipponA/HRTF_sonicom_dataset/Measure_1D_ears/Ear_distances_1D/Ear_distances_left_sonicom.npy')
stats_1D_R=np.load('/home/PhilipponA/HRTF_sonicom_dataset/Measure_1D_ears/Ear_distances_1D/Ear_distances_right_sonicom.npy')
stats_1D=np.concatenate((stats_1D_L,stats_1D_R),axis=0)

stats_1D_L_train=stats_1D_L[:71,:]
stats_1D_R_train=stats_1D_R[:71,:]
stats_1D_train=np.concatenate((stats_1D_L_train,stats_1D_R_train),axis=0)

stats_1D_L_test=stats_1D_L[71:,:]
stats_1D_R_test=stats_1D_R[71:,:]
stats_1D_test=np.concatenate((stats_1D_L_test,stats_1D_R_test),axis=0)

datasets=[stats_1D_L,stats_1D,stats_1D_R,stats_1D_L_train,stats_1D_train,stats_1D_R_train,stats_1D_L_test,stats_1D_test,stats_1D_R_test]


fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20, 20))
axes = axes.flatten()

Names=["Whole stats of the left ears","Whole stats of both ears","Whole stats of the right ears","Train subjects stats of the left ears","Train subjects stats of both ears",
       "Train subjects stats of the right ears","Test subjects stats of the left ears","Test subjects stats of both ears",
       "Test subjects stats of the right ears"]

# Boucle sur les subplots
for i, ax in enumerate(axes):
    data = datasets[i]
    parts = ax.violinplot(data, showmedians=True,showmeans=True ,showextrema=True)
   
    # Personnalisation de l'axe
    ax.set_title(Names[i])
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7])
    ax.set_xticklabels(['cavum concha h', 'cymba concha h', 'cavum concha w','fossa h','pinna h','pinna w','intertragal inci w'])
    ax.set_ylabel('Relative distance')

fig.savefig("Stats 1D.png")

