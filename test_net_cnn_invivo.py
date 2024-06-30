import numpy as np
import os
import CNN.ResNet as resnet
import config

config.config_gpu(0)

################################################################################
#### MODEL
sigma_train, type_te_train = 'mix', 6
Nc_cnn     = 6

suffix_cnn = 'moled_r2_mae_mono_{}_{}'.format(Nc_cnn,sigma_train)

# ------------------------------------------------------------------------------
name_data     = 'invivo'
name_study    = 'Study_30_1'
name_protocol = 'MESE'

# type_te_test = 13
type_te_test = 12
# type_te_test = 11
index_te = np.r_[0,1,3,7,15,31]

################################################################################
#### DATASET
print('='*98)
print('Load dataset ...')
dir_data = os.path.join('data',name_data+'data',name_study,name_protocol+'.npy')

imgs_n = np.load(dir_data)
imgs_n = imgs_n[...,index_te]

data_input = imgs_n

print('-'*98)
print('Test datasets :')
print('Test file name:',dir_data)
print('imgs          :',str(imgs_n.shape))

#################################################################################
##### ResNet model
print('='*98)
print('Load ResNet model ...')
model_epoch = 180
Nb = 4
name_resnet = 'resnet_{}_{}'.format(Nb,suffix_cnn)
dir_model_resnet = os.path.join('model','resnet',name_resnet,'model_{:03d}.h5'.format(model_epoch)) # load weights
print('Model: ',dir_model_resnet)

model_resnet = resnet.ResNet_moled(image_channels=Nc_cnn,output_channel=2,num_block=Nb)
model_resnet.load_weights(filepath=dir_model_resnet)

maps_pred_resnet = model_resnet(data_input)
print('> Output shape: ', maps_pred_resnet.shape)

#################################################################################
print('='*98)
print('Save results ...')
type_experiment = '{}_{}_{}_NLtrn_{}_TEtst_{}_TEtrn_{}'.format(name_data,name_study,name_protocol,sigma_train,type_te_test,type_te_train)
dir_results     = os.path.join('results',type_experiment,name_resnet)

if os.path.exists(dir_results) == False: os.mkdir(dir_results)

print('Save to ', dir_results)
np.save(os.path.join(dir_results,'maps_pred_resnet.npy'),maps_pred_resnet)
print('='*98)
