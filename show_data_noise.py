import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import config
import functions as func

config.config_gpu(0)

###################################################################################################################
print('='*98)
type_task, rescale_para = 'T1mapping', 400.0
# type_task, rescale_para = 'T2mapping', 100.0

# sub, type_data, sigma = 54, 'testingdata', 0.1
sub, type_data, sigma = 45, 'trainingdata', 'mix'

group_tau = 0
id_tau = []
id_sample_show = 80

###################################################################################################################
filenames = tf.io.gfile.glob(os.path.join('data',type_task,type_data,'subject{}_te_{}_sigma_{}*.tfrecords'.format(sub,group_tau,sigma)))
print(filenames)
dataset = tf.data.TFRecordDataset(filenames)\
            .map(func.parse_all)\
            .map(lambda x1,x2,x3,x4,x5,x6,x7: func.extract(x1,x2,x3,x4,x5,x6,x7,\
                id_tau=id_tau,rescale=rescale_para,model_type='test',data_type='mono',task=type_task))
for data in dataset.batch(300).take(1): data = data

imgs_n, maps_gt  = data[0][0].numpy(), data[1].numpy()

print(imgs_n.shape)
print(maps_gt.shape)

Ny,Nx,Nq = imgs_n.shape[-3:]
Np = maps_gt.shape[-1]
###################################################################################################################
colume_width, page_width = 3.5, 7.16
# show patches
fig,axes = plt.subplots(nrows=3,ncols=4,figsize=(colume_width,colume_width/4*3),dpi=600,tight_layout=True)
[ax.set_axis_off() for ax in axes.ravel()]
ax_w = axes[0:2,:].ravel()

if type_task == 'T1mapping':
    for i in range(8):
        ax_w[i].imshow(np.abs(imgs_n[id_sample_show,...,i]),vmin=0.0,vmax=1.5,cmap='gray')
    axes[2,0].imshow(maps_gt[id_sample_show,...,0],vmin=0.0,vmax=2.0,cmap='gray')
    axes[2,1].imshow(maps_gt[id_sample_show,...,1],vmin=0.0,vmax=3.0,cmap='gray')
    axes[2,2].imshow(maps_gt[id_sample_show,...,2],vmin=0.0,vmax=5.0,cmap='jet')

if type_task == 'T2mapping':
    for i in range(8):
        ax_w[i].imshow(imgs_n[id_sample_show,...,i],vmax=1.2,vmin=0.0,cmap='gray')
    axes[2,0].imshow(maps_gt[id_sample_show,...,0],vmin=0.0,vmax=1.2,cmap='gray')
    axes[2,1].imshow(maps_gt[id_sample_show,...,1],vmin=0.0,vmax=3.0,cmap='jet')

plt.savefig(os.path.join('figures',type_task,'data_noise'))

# show histogram of maps
fig,axes = plt.subplots(nrows=1,ncols=3,figsize=(page_width,page_width/3),dpi=600,tight_layout=True)
if type_task == 'T1mapping':
    a,b,c = maps_gt[...,0].flatten(),maps_gt[...,1].flatten(),maps_gt[...,2].flatten()
    axes[0].hist(a[a!=0],bins=100,range=None)
    axes[1].hist(b[b!=0],bins=100,range=None)
    axes[2].hist(c[c!=0],bins=100,range=None)
if type_task == 'T2mapping':
    a,b = maps_gt[...,0].flatten(), maps_gt[...,1].flatten()
    axes[0].hist(a[a!=0],bins=100,range=None)
    axes[1].hist(b[b!=0],bins=100,range=None)
plt.savefig(os.path.join('figures',type_task,'data_noise_his'))


