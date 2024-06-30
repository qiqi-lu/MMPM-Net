'''
Convert noise-free data to noisy data with specific noise levels. (mixed or fixed sigma)
'''
import tensorflow as tf
import os
import numpy as np
import sys

import data_processor as dp
import functions as func
import config

config.config_gpu(0)

###################################################################################################################
type_task,type_para,signed = 'T1mapping', 'T1', 1
# tyep_task,type_para = 'T2mapping', 'T2'


# id_subject, id_tau, num_slice = 45, 0, 140 # training data
# type_data, sigma, num_realization = 'trainingdata', [0.01,0.15], 1 # mixed noise level and the number of realization, for training data


id_subject, id_tau, num_slice = 54, 0, 1   # testing data
type_data, sigma, num_realization = 'testingdata', 0.1, 10        # fixed noise level, for testing data

dir_data = os.path.join('data',type_task,type_data,'subject{}_tau_{}_sigma_0_S_{}.tfrecords'.format(id_subject,id_tau,num_slice))

###################################################################################################################
print('='*98)
if type(sigma) == float: type_sigma = str(sigma)
if type(sigma) == list:  type_sigma = 'mix'

# load data without noise
print('Load noise-free dataset ...')
dataset_filenames = tf.io.gfile.glob(dir_data)
if len(dataset_filenames) == 0:
    print('Cannot find file : '+dir_data)
    sys.exit(1)
else:
    print(dataset_filenames)
num_file = len(dataset_filenames)

print('='*98)
print('Convert to image with noise sigma = ', sigma)
for i in range(num_file):
    print('-'*98)
    # load noise-free data
    filename = dataset_filenames[i]
    dataset  = tf.data.TFRecordDataset(filename).map(func.parse_noise_free)
    N        = func.get_len(dataset)
    print('Processing ',filename,'...')

    for sample in dataset.batch(N).take(1): # get all data in the file.
        print('Converting ...')
        imgs_gt_bi, maps_gt, seg, tau  = sample[0].numpy(), sample[1].numpy(), sample[2].numpy(), sample[3].numpy()
        tau_mono = tau[0]
        print('> Data Info:\n> Image GT: {}\n> Map GT: {}\n> Segmentation: {}\n> TE: {}'
              .format(imgs_gt_bi.shape, maps_gt.shape, seg.shape, tau.shape))
        print('> target te:', tau_mono)
        # generate multi-exponential data
        imgs_gt_bi = np.repeat(imgs_gt_bi,repeats=num_realization,axis=0)           # multiple realization
        imgs_n_bi  = dp.image2noisyimage(imgs=imgs_gt_bi,sigma=sigma,noise_type='Gaussian') # add noise
        # generate mono-exponential data
        imgs_gt_mono = dp.map2image(maps=maps_gt,tau=tau_mono,type_para=type_para,signed=signed)  # parameter mpas to weighted images
        imgs_gt_mono = np.repeat(imgs_gt_mono,repeats=num_realization,axis=0)       # multiple realization
        imgs_n_mono  = dp.image2noisyimage(imgs=imgs_gt_mono,sigma=sigma,noise_type='Gaussian')

        maps_gt = np.repeat(maps_gt,repeats=num_realization,axis=0)                 # multiple realization

        seg = np.repeat(seg,repeats=num_realization,axis=0)                         # multiple realization
        tau = np.repeat(tau,repeats=num_realization,axis=0)                         # multiple realization

        # Save data into TFrecords file.
        imgs_gt_bi  = imgs_gt_bi.astype(np.float32)
        imgs_n_bi   = imgs_n_bi.astype(np.float32)
        imgs_gt_mono= imgs_gt_mono.astype(np.float32)
        imgs_n_mono = imgs_n_mono.astype(np.float32)
        maps_gt     = maps_gt.astype(np.float32)
        seg         = seg.astype(np.float32)
        tau         = tau.astype(np.float32)
        print(imgs_gt_bi.shape,imgs_n_bi.shape,imgs_gt_mono.shape,imgs_n_mono.shape,maps_gt.shape,seg.shape,tau.shape)

        file_name = os.path.join('data',type_task,type_data,'subject{}_tau_{}_sigma_{}_S_{}_N_{}_signed_{}.tfrecords'.\
                                    format(id_subject,id_tau,type_sigma,num_slice,num_realization,signed))
        func.write2TFRecord_noise(imgs_gt_bi=imgs_gt_bi,imgs_n_bi=imgs_n_bi,imgs_gt_mono=imgs_gt_mono,imgs_n_mono=imgs_n_mono,
                            maps_gt=maps_gt,seg=seg,tes=tau,filename=file_name)
print('='*98)
