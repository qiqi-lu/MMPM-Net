import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import config
import functions as func

config.config_gpu(7)

print('='*98)
data_type = 'testingdata'
te_type = 1
sub     = 54
sigma   = 0.1
filenames = tf.io.gfile.glob(os.path.join('data',data_type,'subject{}_te_{}_sigma_{}.tfrecords'.format(sub,te_type,sigma)))
print(filenames)
dataset = tf.data.TFRecordDataset(filenames)\
            .map(func.parse_all)\
            .map(lambda x1,x2,x3,x4,x5,x6,x7: func.extract(x1,x2,x3,x4,x5,x6,x7,\
                num_te=0,rescale=100.0,model_type='test',data_type='mono'))
for data in dataset.batch(25).take(1): data = data

imgs_gt = data[0][0]
maps_gt = data[1]

print(imgs_gt.shape)
print(maps_gt.shape)

Ny,Nx,Nq = imgs_gt.shape[-3:]
Np = maps_gt.shape[-1]

ids= 14
fig,axes = plt.subplots(nrows=2,ncols=Nq,figsize=(Nq,2),dpi=300)
[ax.set_axis_off() for ax in axes.ravel()]
for j in range(Nq):
    axes[0,j].imshow(imgs_gt[ids,...,j],vmax=1.2,vmin=0.0,cmap='gray')
axes[1,0].imshow(maps_gt[ids,...,0],vmin=0.0,vmax=1.2,cmap='gray')
axes[1,1].imshow(maps_gt[ids,...,1],vmin=0.0,vmax=3.0,cmap='jet')
plt.savefig(os.path.join('figures','data_noise.png'))
