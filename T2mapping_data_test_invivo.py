import numpy as np
import matplotlib.pyplot as plt
import os

import functions as func

dir_data_invivo   = os.path.join('data','invivodata')
dir_data_testing  = os.path.join('data','testingdata')

name_study    = 'Study_30_1'
name_protocol = 'MESE'
suffix_data   = 'mat'
offset = 0.0

name_file_data = 'meseslice5.mat'
name_file_mask = 'headmask.mat'

#############################################################################
# Read data
print('='*98)
dir_data = os.path.join(dir_data_invivo,name_study,name_protocol)
print('Read ',dir_data)
if suffix_data == 'mat':
    data = func.read_mat(os.path.join(dir_data,name_file_data))
    imgs = data['mese']
    imgs = np.transpose(imgs,axes=(1,2,0))
    imgs = imgs[:,300:500]
imgs = func.rescale_wimg(imgs,99)
imgs = func.resize_wimg(img=imgs,scale_factor=2.0)

print(imgs.shape)

tes = np.arange(start=10.0,stop=330.0,step=10.0)

#############################################################################
# show
N_row, N_col = 4,8
fig,axes = plt.subplots(nrows=N_row,ncols=N_col,dpi=300,figsize=(N_col*2.,N_row*2.0),constrained_layout=True)
for i,ax in enumerate(axes.ravel()):
    ax.imshow(imgs[...,i],cmap='gray',vmin=0.0,vmax=1.0),ax.set_title(tes[i]),ax.set_axis_off()
plt.savefig(os.path.join('figures','img_invivo'))

#############################################################################
# save npy data
imgs = imgs[np.newaxis]
np.save(file=dir_data,arr=np.array(imgs,dtype=np.float))

# ---------------------------------------------------------------------------
# npy data to raw data
imgs_raw = np.transpose(imgs[0],axes=(2,0,1))
imgs_raw.astype('float64').tofile(dir_data+'.raw')

#############################################################################
