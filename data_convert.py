'''
Convert noise free data to noisy data with specific noise. (mix or fix sigma)
'''
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

import data_processor as dp
import functions as func
import config

config.config_gpu(6)

###################################################################################################################
# type_data = 'trainingdata'
type_data  = 'testingdata'
id_subject, id_te, num_slice = 54, 0, 50
# sigma = [0.01,0.15]
sigma = 0.05
num_realization = 1

dir_data  = os.path.join('data',type_data,'subject{}_te_{}_sigma_0_S_{}.tfrecords'.format(id_subject,id_te,num_slice))

###################################################################################################################
print('='*98)
if type(sigma) == float: type_sigma = str(sigma)
if type(sigma) == list:  type_sigma = 'mix'

# load data without noise
print('Load noise free dataset ...')
dataset_filenames = tf.io.gfile.glob(dir_data)
print(dataset_filenames)
num_file = len(dataset_filenames)

print('='*98)
print('Convert to image with noise sigma = ', sigma)
for i in range(num_file):
    print('-'*98)
    filename = dataset_filenames[i]
    dataset  = tf.data.TFRecordDataset(filename).map(func.parse_noise_free)
    N        = func.get_len(dataset)
    print('Processing ',filename,'[Size:',N,']')

    for sample in dataset.batch(N).take(1):
        print('Converting ...')
        imgs_gt_bi  = sample[0].numpy()
        maps_gt     = sample[1].numpy()
        seg         = sample[2].numpy()
        tes         = sample[3].numpy()

        print('> Data Info:')
        print('> Image GT:', imgs_gt_bi.shape)
        print('> Map GT:', maps_gt.shape)
        print('> Segmentation:', seg.shape)
        print('> TE:', tes.shape)

        tes_mono = tes[0]
        print('> target te:', tes_mono)

        imgs_gt_bi = np.repeat(imgs_gt_bi,repeats=num_realization,axis=0)
        imgs_n_bi  = dp.image2noisyimage(imgs=imgs_gt_bi,sigma=sigma)

        imgs_gt_mono = dp.map2image(maps=maps_gt,tes=tes_mono)
        imgs_gt_mono = np.repeat(imgs_gt_mono,repeats=num_realization,axis=0)
        imgs_n_mono  = dp.image2noisyimage(imgs=imgs_gt_mono,sigma=sigma)

        maps_gt = np.repeat(maps_gt,repeats=num_realization,axis=0)

        seg = np.repeat(seg,repeats=num_realization,axis=0)
        tes = np.repeat(tes,repeats=num_realization,axis=0)

        # Save data into tfrecords file.
        imgs_gt_bi  = imgs_gt_bi.astype(np.float32)
        imgs_n_bi   = imgs_n_bi.astype(np.float32)
        imgs_gt_mono= imgs_gt_mono.astype(np.float32)
        imgs_n_mono = imgs_n_mono.astype(np.float32)
        maps_gt     = maps_gt.astype(np.float32)
        seg         = seg.astype(np.float32)
        tes         = tes.astype(np.float32)
        print(imgs_gt_bi.shape,imgs_n_bi.shape,imgs_gt_mono.shape,imgs_n_mono.shape,maps_gt.shape,seg.shape,tes.shape)

        file_name = os.path.join('data',type_data,'subject{}_te_{}_sigma_{}_S_{}_N_{}.tfrecords'.format(id_subject,id_te,type_sigma,num_slice,num_realization))
        func.write2TFRecord_noise(imgs_gt_bi=imgs_gt_bi,imgs_n_bi=imgs_n_bi,imgs_gt_mono=imgs_gt_mono,imgs_n_mono=imgs_n_mono,
                            maps_gt=maps_gt,seg=seg,tes=tes,filename=file_name)
print('='*98)

###################################################################################################################
# show dataset samples
fig,axes = plt.subplots(nrows=4,ncols=6,figsize=(12,8),dpi=300,constrained_layout=True)
[ax.set_axis_off() for ax in axes.ravel()]
for i in range(3):
    axes[0,i].imshow(imgs_gt_bi[0,...,i],cmap='gray',vmin=0.0,vmax=1.3),   axes[0,i].set_title('GT(bi)(TE'+str(i)+')')  
    axes[0,i+3].imshow(imgs_n_bi[0,...,i]-imgs_n_bi[1,...,i],cmap='gray',vmin=0.0,vmax=0.1), axes[0,i+3].set_title('Rz Diff (TE'+str(i)+')')
    axes[1,i].imshow(imgs_n_bi[0,...,i],cmap='gray',vmin=0.0,vmax=1.3),    axes[1,i].set_title('Noisy (Rz 0) (TE'+str(i)+')')
    axes[1,i+3].imshow(imgs_n_bi[1,...,i],cmap='gray',vmin=0.0,vmax=1.3),  axes[1,i+3].set_title('Noisy (Rz 1) (TE'+str(i)+')')
    axes[2,i].imshow(imgs_gt_mono[0,...,i],cmap='gray',vmin=0.0,vmax=1.3), axes[2,i].set_title('GT(mono)(TE'+str(i)+')')
    axes[2,i+3].imshow(imgs_n_mono[0,...,i]-imgs_n_mono[1,...,i],cmap='gray',vmin=0.0,vmax=0.1),axes[2,i+3].set_title('Rz Diff (TE'+str(i)+')')
    axes[3,i].imshow(imgs_n_mono[0,...,i],cmap='gray',vmin=0.0,vmax=1.3),  axes[3,i].set_title('Noisy (Rz 0) (TE'+str(i)+')')
    axes[3,i+3].imshow(imgs_n_mono[1,...,i],cmap='gray',vmin=0.0,vmax=1.3),axes[3,i+3].set_title('Noisy (Rz 1) (TE'+str(i)+')')
plt.savefig('figures/data_convert')