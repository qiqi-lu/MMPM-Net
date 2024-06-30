import numpy as np
import matplotlib.pyplot as plt
import os

import functions as func

################################################################
dir_data_phantom = os.path.join('data','phantomdata')
dir_data_testing = os.path.join('data','testingdata')

name_study    = 'Study_15_3'
name_protocol = 'MESE'
suffix_data   = 'dcm'
offset = 4.0

################################################################
# read dicom data
dir_data = os.path.join(dir_data_phantom,name_study,name_protocol)
print('Read ',dir_data)
imgs,tes = func.read_dicom(dir_data,suffix=suffix_data)
imgs = imgs-offset
imgs = func.rescale_wimg(imgs,99)
imgs = func.resize_wimg(img=imgs,scale_factor=2.0)

print(imgs.shape)
print(tes)

################################################################
# show
N_row, N_col = 4,8
fig,axes = plt.subplots(nrows=N_row,ncols=N_col,dpi=300,figsize=(N_col*3.,N_row*3.0),constrained_layout=True)
ax = axes.ravel()
for i in range(imgs.shape[-1]):
    ax[i].imshow(imgs[...,i],cmap='gray',vmin=0.0,vmax=1.0),ax[i].set_title(tes[i]),ax[i].set_axis_off()
plt.savefig(os.path.join('figures','img_phantom'))

################################################################
# save npy data
imgs = imgs[np.newaxis]
np.save(file=dir_data,arr=np.array(imgs,dtype=np.float))

# --------------------------------------------------------------
# npy data to raw data
imgs_raw = np.transpose(imgs[0],axes=(2,0,1))
imgs_raw.astype('float64').tofile(dir_data+'.raw')
################################################################
