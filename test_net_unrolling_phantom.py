import numpy as np
import tensorflow as tf
import os

import unrollingnet.ADMMREDNet as admmnet
import unrollingnet.DOPAMINE_T2 as dopamine
import unrollingnet.RIM_T2 as rim
import config

config.config_gpu(5)

################################################################################
sigma_train, type_te_train, rescale = 'mix', 6, 100.0

suffix_admmnet  = 'r2_maemow_6_{}_xfzz_tt_0.001_0.1_11'.format(sigma_train)
# suffix_admmnet  = 'r2_maemow_6_{}_xfzz_tt_0.001_0.1_dncnn_16_lr3'.format(sigma_train)
suffix_dopamine = 'r2_maemow_6_{}_dxx_11'.format(sigma_train)
suffix_rim      = 'r2_mmse_mean_6_{}_l2_0.001_10'.format(sigma_train)

name_data     = 'phantom'
name_study    = 'Study_15_1'

# ------------------------------------------------------------------------------
# name_protocol, tes = 'MESE', np.arange(start=13.,stop=417.,step=13.)

# type_te_test, index_te = 13, np.r_[1:32]

# ------------------------------------------------------------------------------
# name_protocol, tes = 'T2 VTE', np.arange(start=10.,stop=330.,step=10.)
# name_protocol, tes = 'T2SE-TE10-320_0031', np.arange(start=10.,stop=330.,step=10.)

# type_te_test, index_te = 13, np.r_[1:32]

# ------------------------------------------------------------------------------
name_protocol, tes = 'SE', np.array([10.,20.,30.,40.,50.,60.,70.,80.,90.,100.,120.,140.,160.,180.,200.,220.,240.,260.,280.,300.,320.,340.,360.,380.,400.])

# type_te_test, index_te = 1, np.r_[0,1,3,7,12,20]
type_te_test, index_te = 3, np.r_[1,3,5,8,12,20]
# type_te_test, index_te = 11, np.r_[1,3,5,7,9,10,11,12,13,14,15,16,17,18,19,20]

################################################################################
#### DATASET
print('='*98)
print('Load dataset ...')
dir_data = os.path.join('data',name_data+'data',name_study,name_protocol+'.npy')
imgs_n   = np.load(dir_data)
tes      = np.repeat(tes[np.newaxis],imgs_n.shape[0],axis=0)/rescale

print('Data Shape: ', imgs_n.shape)
print('Echo Times: ', tes[0])

################################################################################
print('='*98)
type_experiment = '{}_{}_{}_NLtrn_{}_TEtst_{}_TEtrn_{}'.format(name_data,name_study,name_protocol,sigma_train,type_te_test,type_te_train)
dir_results     = os.path.join('results',type_experiment)
if os.path.exists(dir_results) == False: os.mkdir(dir_results)
print('Save to ', dir_results)

np.save(os.path.join(dir_results,'imgs_n_test.npy'),imgs_n)
np.save(os.path.join(dir_results,'tes_test.npy'),tes)

################################################################################
# Discarding the very first n echo.
imgs_n = imgs_n[...,index_te]
tes    = tes[...,index_te]
data_input = (imgs_n.astype(np.float32),tes.astype(np.float32))

Nc_test = tes.shape[-1]

print('-'*98)
print('{:15}'.format('Test datasets'))
print('{:15}: {}'.format('Test file name',dir_data))
print('{:15}: {}'.format('Imgs',str(imgs_n.shape)))
print('{:15}: {} {}'.format('Tes',str(tes.shape),str(tes[0])))

################################################################################
def predict(model_architecture,model_Nc,model_wieghts,data):
    print('Model: ',model_wieghts)
    # load model weights
    inpt_b   = tf.keras.layers.Input(shape=(None,None,model_Nc))
    inpt_te  = tf.keras.layers.Input(shape=(model_Nc,))
    para_map = model_architecture([inpt_b,inpt_te])
    model    = tf.keras.Model(inputs=[inpt_b,inpt_te],outputs=para_map)
    model.load_weights(filepath=model_wieghts)
    # predict
    maps_pred = model(data)
    print('> Output shape: ', maps_pred.shape)
    return maps_pred

################################################################################
##### ADMMNet model 
print('='*98)
print('Load ADMMNet model ...')
model_epoch = 150
Ns, Nk, Nt, f, path = 15, 5, 1, 3, 1
name_admmnet = 'admmnet_Ns_{}_Nk_{}_Nt_{}_f_{}_path_{}_{}'.format(Ns,Nk,Nt,f,path,suffix_admmnet)
dir_model_admmnet = os.path.join('model','admmnet',name_admmnet,'model_{:03d}.h5'.format(model_epoch))

net_admmnet       = admmnet.ADMMNetm(Ns=Ns,Nk=Nk,Nt=Nt,f=f,path=path,name='ADMMNet')
maps_pred_admmnet = predict(net_admmnet,model_Nc=Nc_test,model_wieghts=dir_model_admmnet,data=data_input)

if os.path.exists(os.path.join(dir_results,name_admmnet)) == False: os.mkdir(os.path.join(dir_results,name_admmnet))
np.save(os.path.join(dir_results,name_admmnet,'maps_pred_admmnet.npy'),maps_pred_admmnet)

#################################################################################
##### DOPAMINE model
print('='*98)
print('Load DOPAMINE model ...')

model_epoch = 150
Ns, path, shared_weight, norm = 10, 1, 0, 0
name_dopamine = 'dopamine_Ns_{}_path_{}_SW_{}_Norm_{}_{}'.format(Ns,path,shared_weight,norm,suffix_dopamine)
dir_model_dopamine = os.path.join('model','dopamine',name_dopamine,'model_{:03d}.h5'.format(model_epoch))

if shared_weight == 0: net_dopamine = dopamine.DOPAMINEm(Ns=Ns,path=path,norm=norm)
if shared_weight == 1: net_dopamine = dopamine.DOPAMINEm_sw(Ns=Ns,path=path,norm=norm)
maps_pred_dopamine = predict(net_dopamine,model_Nc=Nc_test,model_wieghts=dir_model_dopamine,data=data_input)

if os.path.exists(os.path.join(dir_results,name_dopamine)) == False: os.mkdir(os.path.join(dir_results,name_dopamine))
np.save(os.path.join(dir_results,name_dopamine,'maps_pred_dopamine.npy'),maps_pred_dopamine)

#################################################################################
##### RIM model
print('='*98)
print('Load RIM model ...')
model_epoch = 150
Ns, f = 6, 36
name_rim = 'rim_Ns_{}_f_{}_{}'.format(Ns,f,suffix_rim)
dir_model_rim = os.path.join('model','rim',name_rim,'model_{:03d}.h5'.format(model_epoch))

net_rim       = rim.RIMm(Ns=Ns,Np=2,filters=f)
maps_pred_rim = predict(net_rim,model_Nc=Nc_test,model_wieghts=dir_model_rim,data=data_input)

if os.path.exists(os.path.join(dir_results,name_rim)) == False: os.mkdir(os.path.join(dir_results,name_rim))
np.save(os.path.join(dir_results,name_rim,'maps_pred_rim.npy'),maps_pred_rim)

#################################################################################
print('='*98)