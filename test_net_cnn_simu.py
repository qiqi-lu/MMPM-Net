import numpy as np
import tensorflow as tf
import os
import tqdm

import CNN.ResNet as resnet
import functions as func
import config

config.config_gpu(0)

id_te_dict = {0:[0,2,7,15,31,63],
              1:[1,3,7,15,31,63],
              2:[2,5,9,15,31,63],
              3:[3,7,11,17,31,63],
              4:[4,9,13,19,35,63],
              5:[0,2,6,14,30,62],
              6:[5,23],
              7:[5,11,17,23],
              8:[3,7,11,15,19],
              9:[2,5,8,11,14,17],
              10:[4,9,14,19,24,29,34,39],
              11:[1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31],
              12:[1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47],
              13:[1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63]}

################################################################################
sigma_test, type_te_test = '0.05', 12
id_subject, tyep_te_data, num_slice, num_realization = 54, 0, 1, 10

# ------------------------------------------------------------------------------
sigma_train, type_te_train, Nc_cnn = 'mix', 6, 6
suffix_cnn = 'moled_r2_mae_mono_{}_{}_10'.format(Nc_cnn,sigma_train)

if len(id_te_dict[type_te_test]) == Nc_cnn:
    id_te_test = id_te_dict[type_te_test]
else:
    id_te_test = id_te_dict[1]

# ------------------------------------------------------------------------------
num_img = num_slice*num_realization
dir_dataset  = os.path.join('data','testingdata','subject{}_te_{}_sigma_{}_S_{}_N_{}.tfrecords'\
                .format(id_subject,tyep_te_data,sigma_test,num_slice,num_realization))

################################################################################
print('='*98)
print('Load dataset ...')
filenames = tf.io.gfile.glob(dir_dataset)
print(filenames)

dataset_cnn = tf.data.TFRecordDataset(filenames).map(func.parse_all)\
                .map(lambda x1,x2,x3,x4,x5,x6,x7: func.extract(x1,x2,x3,x4,x5,x6,x7,\
                    id_te=id_te_test,rescale=100.0,model_type='test',data_type='multi'))
data_cnn    = dataset_cnn.batch(1).take(num_img)

for sample in dataset_cnn.batch(num_img).take(1):
    imgs_n  = sample[0][0]
    tes     = sample[0][1][0]
    maps_gt = sample[1]
    seg     = sample[2]

print('='*98)
print('Test datasets :')
print('Test file name:',filenames)
print('imgs          :',str(imgs_n.shape))
print('tes           :',str(tes.shape),str(tes))
print('sigma (test)  :',sigma_test)
print('sigma (train) :',sigma_train)

#################################################################################
##### ResNet model
print('='*98)
print('Load ResNet model ...')
model_epoch = 150
Nb = 4
name_resnet = 'resnet_{}_{}'.format(Nb,suffix_cnn)
dir_model_resnet = os.path.join('model','resnet',name_resnet,'model_{:03d}.h5'.format(model_epoch))
print('Model: ',dir_model_resnet)

model_resnet = resnet.ResNet_moled(image_channels=Nc_cnn,output_channel=2,num_block=Nb)
model_resnet.load_weights(filepath=dir_model_resnet)
# model_resnet.summary()

#################################################################################
# Predict
maps_pred_resnet = []
pbar = tqdm.tqdm(desc='CNN',total=num_img)
for sample in data_cnn:
    maps_pred_resnet.append(model_resnet(sample[0][0]))
    pbar.update(1)
pbar.close()
maps_pred_resnet = np.concatenate(maps_pred_resnet,axis=0)
print('> Output shape: ', maps_pred_resnet.shape)

#################################################################################
print('='*98)
print('Save results ...')
type_experiment = 'NLtst_{}_NLtrn_{}_TEtst_{}_TEtrn_{}_S_{}_N_{}'\
                .format(sigma_test,sigma_train,type_te_test,type_te_train,num_slice,num_realization)
dir_results     = os.path.join('results',type_experiment)
if os.path.exists(dir_results) == False: os.mkdir(dir_results)
print('Save to ', dir_results)

if os.path.exists(os.path.join(dir_results,name_resnet)) == False: os.mkdir(os.path.join(dir_results,name_resnet))
np.save(os.path.join(dir_results,name_resnet,'maps_pred_resnet.npy'),maps_pred_resnet)
print('='*98)
#################################################################################