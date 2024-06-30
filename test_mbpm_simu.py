# Parameter mapping using model based parameter mapping.
import matplotlib.pyplot as plt
import numpy as np
import os
import model_based_para_map as mbpm

##########################################################################################
sigma_test, type_te_test   = 0.05, 6
num_slice, num_realization = 50, 1

# ----------------------------------------------------------------------------------------
sigma_train, type_te_train = 'mix', 6
type_application = 'NLtst_{}_NLtrn_{}_TEtst_{}_TEtrn_{}_S_{}_N_{}'.format(sigma_test,sigma_train,type_te_test,type_te_train,num_slice,num_realization)
dir_results      = os.path.join('results',type_application)

##########################################################################################
print('='*98)
print('Parameter mapping using model based methods.')
print('Load weighted images ...')
imgs_n  = np.load(os.path.join(dir_results,'imgs_n_test.npy'))
maps_gt = np.load(os.path.join(dir_results,'maps_gt_test.npy'))
tes     = np.load(os.path.join(dir_results,'tes_test.npy'))
N,Ny,Nx,Nq = imgs_n.shape

print('> ',type_application)
print('> Testing data size: ',imgs_n.shape)
print('> TE: ',tes)
##########################################################################################
# MLE
print('='*98)
print('MLE ...')
maps_mle = mbpm.pixel_wise_mapping(s=imgs_n.astype(np.float64),tau=tes.astype(np.float64),method='nlls',model_type='mono_exp_r2',pbar_leave=True,ac=8)
np.save(file= os.path.join(dir_results,'maps_mle'),arr=maps_mle)

##########################################################################################
# MAP
print('='*98)
print('MAP ...')

##########################################################################################
# Show results
maps_mle = np.load(file= os.path.join(dir_results,'maps_mle')+'.npy')
print(maps_mle.shape)

Nrow,Ncol = 4,2
idx = 0
fig,axes = plt.subplots(nrows=Nrow,ncols=Ncol,figsize=(2.5*Ncol,2.5*Nrow),dpi=300,tight_layout=True)
axes[0,0].imshow(maps_gt[idx,...,0],cmap='gray',vmin=0.0,vmax=1.3)
axes[2,0].imshow(maps_gt[idx,...,1],cmap='jet',vmin=0.0,vmax=2.0)
axes[0,1].imshow(maps_mle[idx,...,0],cmap='gray',vmin=0.0,vmax=1.3)
axes[2,1].imshow(maps_mle[idx,...,1],cmap='jet',vmin=0.0,vmax=2.0)
axes[1,1].imshow(np.abs(maps_mle[idx,...,0]-maps_gt[idx,...,0]),cmap='gray',vmin=0.0,vmax=0.13)
axes[3,1].imshow(np.abs(maps_mle[idx,...,1]-maps_gt[idx,...,1]),cmap='jet',vmin=0.0,vmax=0.4)
plt.savefig(os.path.join('figures','maps_mle'))
##########################################################################################
print('='*98)