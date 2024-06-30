import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import time

import unrollingnet.ADMMREDNet as admmnet
import unrollingnet.DOPAMINE as dopamine
import functions as func
import config

config.config_gpu(6)
################################################################################
id_te_dict = {1:[1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63],
              2:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],
              3:[1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31],
              4:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
              5:[0,2,6,14,30,62],
              6:[1,3,7,15,31,63]}

sigma_test   = '0.15'
type_te_test = 1
id_te_test   = id_te_dict[type_te_test]

id_subject,tyep_te_data,num_slice,num_realization = 54,0,1,10

dir_dataset  = os.path.join('data','testingdata','subject{}_te_{}_sigma_{}_S_{}_N_{}.tfrecords'\
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
    tes     = sample[0][1]
    maps_gt = sample[1]
    seg     = sample[2]

print('='*98)
print('Test datasets :')
print('Test file name:',filenames)
print('imgs          :',str(imgs_n.shape))
print('tes           :',str(tes[0].shape),str(tes[0]))
print('sigma (test)  :',sigma_test)

################################################################################
# ADMM-REDNet
name = 'var'
rho   = tf.Variable(initial_value=0.0,trainable=False,name=name+'_rho',constraint=tf.keras.constraints.NonNeg())
sigma = tf.Variable(initial_value=1.0,trainable=False,name=name+'_sigma',constraint=tf.keras.constraints.NonNeg())
reconblock  = admmnet.ReconBlock(name='recon')

b   = imgs_n
tes = tes

# x = tf.math.add(tf.math.multiply(b[...,0:2],[0.0,0.0]),1.0)
m0 = tf.math.reduce_max(b,axis=-1)
p2 = tf.math.add(tf.math.multiply(b[...,0],0.0),1.0)
x  = tf.stack([m0,p2],axis=-1)

x  = admmnet.range_constaint(x,m0_max=3.0,p2_max=10.0)
z  = x
beta = x-z

xm   = []
xm.append(x)
Nk = 300

t = time.time()
for _ in range(Nk):
    x = reconblock(x,z,beta,rho,sigma,b,tes) 
    x = admmnet.range_constaint(x,m0_max=3.0,p2_max=10.0)
xm.append(x)
print('>> time: ',time.time()-t)

################################################################################
# DOPAMINE
# Jr = dopamine.Jr(name='Jr')

# t = time.time()
# for _ in range(Nk):
#     gx = Jr(x,b,tes)
#     x  = tf.math.subtract(x,tf.math.multiply(2.0,gx))
#     x = admmnet.range_constaint(x,m0_max=3.0,p2_max=10.0)
# print('>> time: ',time.time()-t)
# xm.append(x)
# xm = tf.stack(xm)
# xm = xm[-10:]

######################################################################
xm = tf.stack(xm)
print(xm.shape)
idx = 0
map_est  = xm.numpy()
map_gt   = maps_gt.numpy()
N = xm.shape[0]
fig,axes = plt.subplots(nrows=4,ncols=N+1,figsize=(N*3,4*3),dpi=300,tight_layout=True)
[ax.set_axis_off() for ax in axes.ravel()]
axes[0,-1].imshow(map_gt[idx,...,0],cmap='gray',vmin=0.0,vmax=1.2)
axes[2,-1].imshow(map_gt[idx,...,1],cmap='jet',vmin=0.0,vmax=2.0)
for i in range(N):
    axes[0,i].imshow(map_est[i,idx,...,0],cmap='gray',vmin=0.0,vmax=1.2)
    axes[0,i].set_title('Iter '+str(i))
    axes[1,i].imshow((map_est[i,idx,...,0]-map_gt[idx,...,0])/map_gt[idx,...,0],cmap='seismic',vmin=-0.5,vmax=0.5)
    axes[2,i].imshow(map_est[i,idx,...,1],cmap='jet',vmin=0.0,vmax=2.0)
    axes[3,i].imshow((map_est[i,idx,...,1]-map_gt[idx,...,0])/map_gt[idx,...,0],cmap='seismic',vmin=-2.0,vmax=2.0)
plt.savefig('figures/reconblock')
######################################################################
