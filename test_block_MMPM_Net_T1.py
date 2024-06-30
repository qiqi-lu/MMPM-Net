import os
import tensorflow as tf
import config
import functions as func
import functions_data as dfunc
import unrollingnet.auxiliary_functions as afunc
import unrollingnet.MMPM_Net_T1 as net
import numpy as np
import matplotlib.pyplot as plt

config.config_gpu(5)

tf.config.run_functions_eagerly(False)

b   = np.load(file=os.path.join('figures','T1mapping','b.npy'))
tau = np.load(file=os.path.join('figures','T1mapping','tau.npy'))
gt  = np.load(file=os.path.join('figures','T1mapping','gt.npy'))
print(gt.shape)
print(b.shape)
print(tau.shape)

print('='*50)
#######################################################################################
# Test reconstruction block

range_cons = afunc.range_constaint_R1(A_max=3.0,B_max=6.0,R1_max=50.0)

A  = tf.math.reduce_max(tf.math.abs(b),axis=-1)
B  = tf.math.multiply(2.0,A)
R1 = tf.math.multiply(tf.ones_like(A),1.0)
x0  = tf.stack([A,B,R1],axis=-1)
x0  = range_cons(x0)
z  = x0
beta = tf.zeros_like(x0)

recon = net.ReconBlock(signed=True)

x = x0
for _ in range(10):
    x = recon(x,z,beta,0.0,1.0,b,tau) 
    x = range_cons(x)

page_width = 7.16
fig,axes = plt.subplots(nrows=2,ncols=9,figsize=(page_width,page_width/9*2),dpi=600,tight_layout=True)
[ax.set_axis_off() for ax in axes.ravel()]
for i in range(8):
    axes[0,i].imshow(np.abs(b[0,:,:,i]),cmap='gray',vmin=0.0,vmax=2.0)
axes[1,0].imshow(gt[0,:,:,0],cmap='gray',vmin=0.0,vmax=2.0)
axes[1,1].imshow(gt[0,:,:,1],cmap='gray',vmin=0.0,vmax=3.0)
axes[1,2].imshow(gt[0,:,:,2],cmap='jet',vmin=0.0,vmax=3.0)
axes[1,3].imshow(x0[0,:,:,0],cmap='gray',vmin=0.0,vmax=2.0)
axes[1,4].imshow(x0[0,:,:,1],cmap='gray',vmin=0.0,vmax=3.0)
axes[1,5].imshow(x0[0,:,:,2],cmap='jet',vmin=0.0,vmax=3.0)
axes[1,6].imshow(x[0,:,:,0],cmap='gray',vmin=0.0,vmax=2.0)
axes[1,7].imshow(x[0,:,:,1],cmap='gray',vmin=0.0,vmax=3.0)
axes[1,8].imshow(x[0,:,:,2],cmap='jet',vmin=0.0,vmax=3.0)
plt.savefig(os.path.join('figures','T1mapping','block_test'))

print('='*50)
#######################################################################################
Nc = 8
Ns = 10 # num of stage 
Nk = 5  # num of iteration in reoncstruction block
Nt = 1  # num of iteration in auxiliary variable update block
ini_lam, ini_rho = 0.001, 0.1
signed = 1
Np, sep, f = 3, 0, 3

#######################################################################################
nn = net.MMPM_Net(Ns=Ns,Nk=Nk,Nt=Nt,Np=Np,signed=signed,sep=sep,f=f,ini_lam=ini_lam,ini_rho=ini_rho,test_mode=False,name='MMPM_Net')
inpt_b   = tf.keras.layers.Input(shape=(None,None,Nc),name='images')
inpt_tau = tf.keras.layers.Input(shape=(Nc,),name='tes')
para_map = nn([inpt_b,inpt_tau])
model    = tf.keras.Model(inputs=[inpt_b,inpt_tau],outputs=para_map)

model.summary()

out = model((b,tau))

print('end')