import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import config
import functions as func

config.config_gpu(6)

##########################################################################################
id_te, sub, num_slice = 1, 45, 140
data_type = 'trainingdata'
# data_type = 'testingdata'
id_slice_show = 80

filenames = tf.io.gfile.glob(os.path.join('data',data_type,'subject{}_te_{}_sigma_0_S_{}.tfrecords'.format(sub,id_te,num_slice)))
dataset   = tf.data.TFRecordDataset(filenames).map(func.parse_noise_free)
for d in dataset.batch(num_slice).take(1): data = d

imgs_gt = data[0]
maps_gt = data[1]
seg_gt  = data[2]
tes_gt  = data[3]

print(imgs_gt.shape)
print(maps_gt.shape)
print(seg_gt.shape)
print(tes_gt.shape)

tes = tes_gt[0]
Nq  = imgs_gt.shape[-1]
Np  = maps_gt.shape[-1]
Nm  = seg_gt.shape[-1]

##########################################################################################
id_te_show = [0,1,2,3,4,5,-1]
# ----------------------------------------------------------------------------------------
fig,axes = plt.subplots(nrows=6,ncols=6,figsize=(6,6),dpi=300)
[ax.set_axis_off() for ax in axes.ravel()]
for i,ax in enumerate(axes.ravel()):
    ax.imshow(np.nan_to_num(1000.0/maps_gt[id_slice_show+i,...,1]),vmin=0.0,vmax=20.0,cmap='jet')
    # ax.imshow(maps_gt[id_slice_show+i,...,1],vmin=0.0,vmax=300.0,cmap='jet')
plt.savefig(os.path.join('figures','data_sigma_0.png'))

# ----------------------------------------------------------------------------------------

##########################################################################################
print('='*98)
