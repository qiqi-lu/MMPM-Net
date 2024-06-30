# Parameter mapping using model based parameter mapping.
import matplotlib.pyplot as plt
import numpy as np
import os
import model_based_para_map as mbpm

##########################################################################################
name_data     = 'invivo'
name_study    = 'Study_30_1'
name_protocol = 'MESE'

# ----------------------------------------------------------------------------------------
type_te_test = 13
index_te = np.r_[0:32]

# type_te_test = 12
# index_te = np.r_[0:24]

# type_te_test = 11
# index_te = np.r_[0:16]

##########################################################################################
sigma_train, type_te_train = 'mix', 6
type_application = '{}_{}_{}_NLtrn_{}_TEtst_{}_TEtrn_{}'.format(name_data,name_study,name_protocol,sigma_train,type_te_test,type_te_train)
dir_results      = os.path.join('results',type_application)

print('='*98)
print('==== Parameter mapping using model based methods ====')
print('Load weighted images ...')
imgs_n  = np.load(os.path.join(dir_results,'imgs_n_test.npy'))
tes     = np.load(os.path.join(dir_results,'tes_test.npy'))[0]
N,Ny,Nx,Nq = imgs_n.shape

imgs_n = imgs_n[...,index_te]
tes    = tes[...,index_te]

print('> ',type_application)
print('> Testing data size: ',imgs_n.shape)
print('> TE: ',tes)
##########################################################################################
# MLE
print('='*98)
print('MLE ...')
path_file = os.path.join(dir_results,'maps_mle')
maps_mle = mbpm.pixel_wise_mapping(s=imgs_n.astype(np.float64),tau=tes.astype(np.float64),method='nlls',model_type='mono_exp_r2',pbar_leave=True,ac=4)
np.save(file=path_file,arr=maps_mle)

maps_mle = np.load(file=path_file+'.npy')
print(maps_mle.shape)

##########################################################################################
# Show results
Nrow,Ncol = 1,2
idx = 0
fig,axes = plt.subplots(nrows=Nrow,ncols=Ncol,figsize=(2.5*Ncol,2.5*Nrow),dpi=300,tight_layout=True)
axes[0].imshow(maps_mle[idx,...,0],cmap='gray',vmin=0.0,vmax=1.3)
axes[1].imshow(maps_mle[idx,...,1],cmap='jet',vmin=0.0,vmax=2.0)
plt.savefig(os.path.join('figures','maps_mle_'+name_data))
print('='*98)

##########################################################################################
