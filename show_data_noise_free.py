# show data witout noise.
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import config
import functions as func

config.config_gpu(0)

##########################################################################################
# id_tau, sub, num_slice, id_sample_show = 0, 54, 50, 25 # T2 mapping testing data
id_tau, sub, num_slice, id_sample_show = 0, 45, 140, 80  # T1 mapping training data

type_task = 'T1mapping'
# type_task = 'T2mapping'

type_data = 'trainingdata'
# type_data = 'testingdata'

##########################################################################################
filenames = tf.io.gfile.glob(os.path.join('data',type_task,type_data,'subject{}_te_{}_sigma_0_S_{}.tfrecords'.format(sub,id_tau,num_slice)))
dataset   = tf.data.TFRecordDataset(filenames).map(func.parse_noise_free)
for d in dataset.batch(num_slice).take(1): data = d

imgs_gt,maps_gt,seg_gt,tau_gt = data[0],data[1],data[2],data[3]

print(imgs_gt.shape)
print(maps_gt.shape)
print(seg_gt.shape)
print(tau_gt.shape)

tes, Nq, Np, Nm = tau_gt[0], imgs_gt.shape[-1], maps_gt.shape[-1], seg_gt.shape[-1]

##########################################################################################
colume_width, page_with = 3.5, 7.16
# ----------------------------------------------------------------------------------------
fig,axes = plt.subplots(nrows=3,ncols=4,figsize=(colume_width,colume_width/4*3),dpi=600,tight_layout=True)
[ax.set_axis_off() for ax in axes.ravel()]
ax_w = axes[0:2,:].ravel()

if type_task == 'T1mapping':
    for i in range(8):
        ax_w[i].imshow(np.abs(imgs_gt[id_sample_show,:,:,i]),vmin=0.0,vmax=1.5,cmap='gray')
    axes[2,0].imshow(maps_gt[id_sample_show,:,:,0],vmin=0.0,vmax=2.0,cmap='gray')
    axes[2,1].imshow(maps_gt[id_sample_show,:,:,1],vmin=0.0,vmax=3.0,cmap='gray')
    axes[2,2].imshow(maps_gt[id_sample_show,...,2],vmin=0.0,vmax=2000.0,cmap='jet')
    axes[2,3].imshow(seg_gt[id_sample_show,...,0],vmin=0,vmax=10,cmap='hot')

if type_task == 'T2mapping':
    id_tau_show = [0,1,2,3,4,-3,-2,-1]
    for i,tau in enumerate(id_tau_show):
        ax_w[i].imshow(np.abs(imgs_gt[id_sample_show,:,:,tau]),vmin=0.0,vmax=1.5,cmap='gray')
    axes[2,0].imshow(maps_gt[id_sample_show,:,:,0],vmin=0.0,vmax=2.0,cmap='gray')
    axes[2,1].imshow(maps_gt[id_sample_show,:,:,1],vmin=0.0,vmax=1500.0,cmap='jet')
    axes[2,3].imshow(seg_gt[id_sample_show,...,0],vmin=0,vmax=10,cmap='hot')
plt.savefig(os.path.join('figures',type_task,'data_sigma_0.png'))

# ----------------------------------------------------------------------------------------

##########################################################################################
print('='*98)
