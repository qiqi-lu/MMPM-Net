import numpy as np
import tensorflow as tf
import os
import tqdm

import unrollingnet.ADMMREDNet as admmnet
import unrollingnet.DOPAMINE as dopamine
import unrollingnet.RIM as rim
import functions as func
import config

config.config_gpu(0)

################################################################################
id_te_dict = {0: [0,2,7,15,31,63],
              1: [1,3,7,15,31,63],
              2: [2,5,9,15,31,63],
              3: [3,7,11,17,31,63],
              4: [4,9,13,19,35,63],
              5: [0,2,6,14,30,62],
              6: [5,23],
              7: [5,11,17,23],
              8: [3,7,11,15,19],
              9: [2,5,8,11,14,17],
              10:[4,9,14,19,24,29,34,39],
              11:[1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31],
              12:[1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47],
              13:[1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63]}

################################################################################
sigma_test, type_te_test = '0.05', 1
id_subject, tyep_te_data, num_slice, num_realization = 54, 0, 1, 10

# ------------------------------------------------------------------------------
id_te_test = id_te_dict[type_te_test]
sigma_train, type_te_train = 'mix', 6

suffix_admmnet  = 'r2_maemow_6_{}_xfzz_tt_0.001_0.1_11'.format(sigma_train)
# suffix_admmnet  = 'r2_maemow_6_{}_xfzz_tt_0.001_0.1_dncnn_10_lr3'.format(sigma_train)
suffix_dopamine = 'r2_maemow_6_{}_dxx_11'.format(sigma_train)
suffix_rim      = 'r2_mmse_mean_6_{}_l2_0.001'.format(sigma_train)

# ------------------------------------------------------------------------------
dir_dataset = os.path.join('data','testingdata','subject{}_te_{}_sigma_{}_S_{}_N_{}.tfrecords'\
                .format(id_subject,tyep_te_data,sigma_test,num_slice,num_realization))
num_img = num_slice*num_realization

################################################################################
print('='*98)
print('Load dataset ...')
filenames = tf.io.gfile.glob(dir_dataset)
print(filenames)

dataset_unrolling = tf.data.TFRecordDataset(filenames).map(func.parse_all)\
                    .map(lambda x1,x2,x3,x4,x5,x6,x7: func.extract(x1,x2,x3,x4,x5,x6,x7,\
                        id_te=id_te_test,rescale=100.0,model_type='test',data_type='multi'))

for sample in dataset_unrolling.batch(num_img).take(1):
    imgs_n  = sample[0][0]
    tes     = sample[0][1][0]
    maps_gt = sample[1]
    seg     = sample[2]

Nc_test = imgs_n.shape[-1]
data_unrolling = dataset_unrolling.batch(1).take(num_img)

print('-'*98)
print('Test datasets :')
print('Test file name:',filenames)
print('imgs          :',str(imgs_n.shape))
print('tes           :',str(tes.shape),str(tes))
print('sigma (test)  :',sigma_test)
print('sigma (train) :',sigma_train)

################################################################################
def predict(model_architecture,model_Nc,model_wieghts,data,retrun_model=False):
    print('Model: ',model_wieghts)
    # load model weights
    inpt_b   = tf.keras.layers.Input(shape=(None,None,model_Nc))
    inpt_te  = tf.keras.layers.Input(shape=(model_Nc,))
    para_map = model_architecture([inpt_b,inpt_te])
    model    = tf.keras.Model(inputs=[inpt_b,inpt_te],outputs=para_map)
    model.load_weights(filepath=model_wieghts)
    # model.summary()

    maps_pred = []
    pbar = tqdm.tqdm(total=num_img,desc='Prediction')
    for sample in data:
        maps_pred.append(model(sample[0]))
        pbar.update(1)
    pbar.close()
    maps_pred = np.concatenate(maps_pred,axis=1)
    print('> Output shape: ', maps_pred.shape)
    if retrun_model == True  : return maps_pred, model
    if retrun_model == False : return maps_pred

################################################################################
print('='*98)
print('Save results ...')
type_experiment = 'NLtst_{}_NLtrn_{}_TEtst_{}_TEtrn_{}_S_{}_N_{}'\
                    .format(sigma_test,sigma_train,type_te_test,type_te_train,num_slice,num_realization)
dir_results     = os.path.join('results',type_experiment)
if os.path.exists(dir_results) == False: os.mkdir(dir_results)
print('Save to ', dir_results)
np.save(os.path.join(dir_results,'maps_gt_test.npy'),maps_gt)
np.save(os.path.join(dir_results,'imgs_n_test.npy'),imgs_n)
np.save(os.path.join(dir_results,'seg_test.npy'),seg)
np.save(os.path.join(dir_results,'tes_test.npy'),tes)

################################################################################
##### ADMMNet model 
print('='*98)
print('Load ADMMNet model ...')
model_epoch = 150
Ns, Nk, Nt, f, path = 15, 5, 1, 3, 1
name_admmnet = 'admmnet_Ns_{}_Nk_{}_Nt_{}_f_{}_path_{}_{}'.format(Ns,Nk,Nt,f,path,suffix_admmnet)
dir_model_admmnet = os.path.join('model','admmnet',name_admmnet,'model_{:03d}.h5'.format(model_epoch))

net_admmnet       = admmnet.ADMMNetm(Ns=Ns,Nk=Nk,Nt=Nt,f=f,path=path,name='ADMMNet')
maps_pred_admmnet = predict(net_admmnet,model_Nc=Nc_test,model_wieghts=dir_model_admmnet,data=data_unrolling)

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
maps_pred_dopamine = predict(net_dopamine,model_Nc=Nc_test,model_wieghts=dir_model_dopamine,data=data_unrolling)

if os.path.exists(os.path.join(dir_results,name_dopamine)) == False: os.mkdir(os.path.join(dir_results,name_dopamine))
np.save(os.path.join(dir_results,name_dopamine,'maps_pred_dopamine.npy'),maps_pred_dopamine)

#################################################################################
##### RIM model
print('='*98)
print('Load RIM model ...')
model_epoch = 150
Ns, f  = 6, 36
name_rim = 'rim_Ns_{}_f_{}_{}'.format(Ns,f,suffix_rim)
dir_model_rim = os.path.join('model','rim',name_rim,'model_{:03d}.h5'.format(model_epoch))

net_rim       = rim.RIMm(Ns=Ns,Np=2,filters=f)
maps_pred_rim = predict(net_rim,model_Nc=Nc_test,model_wieghts=dir_model_rim,data=data_unrolling)

if os.path.exists(os.path.join(dir_results,name_rim)) == False: os.mkdir(os.path.join(dir_results,name_rim))
np.save(os.path.join(dir_results,name_rim,'maps_pred_rim.npy'),maps_pred_rim)

#################################################################################
print('='*98)
print(dir_results)