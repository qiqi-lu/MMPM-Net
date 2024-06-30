import numpy as np
import os
import matplotlib.pyplot as plt
import functions as func
import scipy.io as sio
import data_processor as dp

# load MAP datasets
folder = os.path.join('data','T1mapping','invivo_MAP')
TIs  = sio.loadmat(os.path.join(folder,'TIs.mat'))['TIs']
data = sio.loadmat(os.path.join(folder,'image.mat'))['data']

# T1* mapping (matlab)
A   = sio.loadmat(os.path.join(folder,'A.mat'))['A']
B   = sio.loadmat(os.path.join(folder,'B.mat'))['B']
T1s = sio.loadmat(os.path.join(folder,'T1s.mat'))['T1s']
T1  = (B/A-1.0)*T1s

# T1* mapping (python)
# TIs = np.squeeze(TIs)
# data = data[np.newaxis]
# data = np.transpose(data,axes=(0,2,3,1))
# para_map = dp.image2map(imgs=np.abs(data),tau=TIs,fitting_model='T1_three_para_magn',signed=False,parameter_type='time',
#                         algorithm='NLLS',pbar_disable=False,ac=8)
# np.save(os.path.join('figures','MAP_maps_signed.npy'),arr=para_map)
# np.save(os.path.join('figures','MAP_maps_unsigned.npy'),arr=para_map)

para_maps = np.load(os.path.join('figures','MAP_maps_unsigned.npy'))
# para_maps = np.load(os.path.join('figures','MAP_maps_signed.npy'))
para_maps = np.rot90(para_maps,k=1,axes=(1,2))
T1_map = (para_maps[0,...,1]/para_maps[0,...,0]-1.0)*para_maps[0,...,2]

# compare results from matlab and python
fig,axes = plt.subplots(nrows=3,ncols=4,dpi=600,figsize=(7.16,7.16/4*3),tight_layout=True)
[ax.set_axis_off() for ax in axes.ravel()]
axes[0,0].imshow(para_maps[0,...,0],vmin=0.0,vmax=1.0,cmap='gray')
axes[0,1].imshow(para_maps[0,...,1],vmin=0.0,vmax=2.0,cmap='gray')
axes[0,2].imshow(para_maps[0,...,2],vmin=0.0,vmax=5000.0,cmap='jet')
axes[0,3].imshow(T1_map,vmin=0.0,vmax=5000.0,cmap='jet')

axes[1,0].imshow(A,vmin=0.0,vmax=1.0,cmap='gray')
axes[1,1].imshow(B,vmin=0.0,vmax=2.0,cmap='gray')
axes[1,2].imshow(T1s,vmin=0.0,vmax=5000.0,cmap='jet')
axes[1,3].imshow(T1,vmin=0.0,vmax=5000.0,cmap='jet')

axes[2,0].imshow((para_maps[0,...,0]-A)/A,vmin=0.0,vmax=0.01,cmap='gray')
axes[2,1].imshow((para_maps[0,...,1]-B)/B,vmin=0.0,vmax=0.01,cmap='gray')
axes[2,2].imshow((para_maps[0,...,2]-T1s)/T1s,vmin=0.0,vmax=0.01,cmap='jet')
axes[2,3].imshow((T1_map-T1)/T1,vmin=0.0,vmax=0.01,cmap='jet')

plt.savefig(os.path.join('figures','para_maps'))
