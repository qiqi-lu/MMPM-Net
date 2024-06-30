import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import config
import functions as func
import unrollingnet.ADMMNetFP as admmnet
import unrollingnet.DOPAMINE_t2 as dopamine
import unrollingnet.RIM_t2 as rim
import os
import time
import h5py

config.config_gpu(7)

def LogLinearN(b,tes,n=3):
    """
    Log-Linear curve fitting Method.
    Perfrom a pixel-wise linear fit of the decay curve after a log transofrmation (using the first n data point).
    #### AGUMENTS
    - TEs : Echo Time (ms)

    #### RETURN
    - map : parameter maps [M0, T2].
    """
    b = tf.math.abs(b)
    x = tes[:,tf.newaxis,tf.newaxis,0:n] 
    y = tf.math.log(b[...,0:n]+1e-5) # log signal

    x_mean = tf.math.reduce_mean(x,axis=-1,keepdims=True)
    y_mean = tf.math.reduce_mean(y,axis=-1,keepdims=True)
    w = tf.math.reduce_sum((x-x_mean)*(y-y_mean),axis=-1,keepdims=True)/tf.math.reduce_sum((x-x_mean)**2,axis=-1,keepdims=True)
    c = y_mean-w*x_mean
    t2  = 1.0/-w[...,0]
    m0  = tf.math.exp(c)[...,0]
    map = tf.stack((m0,t2),axis=-1)
    map = tf.where(map<0.0,0.0,map)
    map = tf.where(map>3000.0,3000.0,map)
    return map

train_filenames = tf.io.gfile.glob(os.path.join('data','trainingdata','*.tfrecords'))
print(train_filenames)
dataset = tf.data.TFRecordDataset(train_filenames).map(func._parse_function_t2_mono)

# MODEL
Ns, Nk, Nt, Np, path = 5, 1, 1, 1, 1
# nn = admmnet.ADMMNet(Ns=Ns, Nk=Nk, Nt=Nt, f=3,q_trainable=True,path=2)
nn = admmnet.ADMMNetm(Ns=Ns, Nk=Nk, Nt=Nt, f=3,q_trainable=True,path=2,name='ADMMNet')
# nn = dopamine.DOPAMINE(Ns=Ns)
# nn = rim.RIMm(Ns=Ns,Np=Np)
# nn = admmnet_auto.ADMMNet_auto(Ns=Ns, Nk=Nk, Nt=Nt,L=128,f=3)
# nn = admmnet2.ADMMNet(Ns=Ns, Nk=Nk, Nt=Nt,f=3,q_trainable=True)


for sample in dataset.batch(100).take(1):
    # print(sample[0][0].shape)
    # print(sample[0][1].shape)
    inputs = sample[0]
    # print('in:',np.any(np.isnan(inputs[0])))
    para_map = nn(inputs)
    
    # x = LogLinearN(b=inputs[0],tes=inputs[1],n=3)
    # print(para_map.shape)
    # func.display_image(inputs[0],filename='figures/weights.png')
    # func.display_func(sample[0][1],filename='figures/func.png')
    # print('out:',np.any(np.isnan(para_map)))
    # if tf.math.reduce_max(sample[1]).numpy()>20.0 or tf.math.reduce_min(sample[1]).numpy()<0.0:
        # print(tf.math.reduce_max(sample[1]).numpy(),tf.math.reduce_min(sample[1]).numpy())
    
    # if tf.math.reduce_max(inputs[0]).numpy()>2.0:
    #     print(tf.math.reduce_max(inputs[0]).numpy(),tf.math.reduce_min(inputs[0]).numpy())

variable_names = [v.name for v in nn.weights]
# print("variables: {}".format(variable_names))
for i in variable_names:
    print(i)

# def read_hdf5(path):

#     weights = {}
    
#     keys = []
#     with h5py.File(path, 'r') as f: # open file
#         f.visit(keys.append) # append all keys to list
#         for key in keys:
#             if ':' in key: # contains data if ':' in key
#                 print(f[key].name)
#                 weights[f[key].name] = f[key].value
#     return weights
# file_name ='model/admmnet/admmnet_Ns_10_Nk_3_Nt_3_f_3_PLU_0_path_2_t2_mae_multi_100_1_1_neg_relu_sp_mono/model_010.h5'
# read_hdf5(file_name)