import tensorflow as tf
import model_modl as mm
import numpy as np
import config
import helper
import os
config.config_gpu(3)

# DATA
# noise_type='Rician'
noise_type='Gaussian'
sg=10 # nosie standard deviation

data_dir  = os.path.join('data',noise_type,str(sg))

# Make training data
tes = tf.constant([0.93, 2.27, 3.61, 4.95, 6.29, 7.63, 8.97, 10.4, 11.8, 13.2, 14.6, 16.0])

print('Load Saved Data (training)...')
imgs=np.load(os.path.join(data_dir,'wImg.npy'))[0:128]
imgs_n=np.load(os.path.join(data_dir,'wImgN.npy'))[0:128]
maps=np.load(os.path.join(data_dir,'ParaRef.npy'))[0:128]

x_train_b =imgs_n
x_train_x0=helper.LogLinear2(x_train_b,tes)
y_train_x =maps

b=x_train_b
x0=x_train_x0

# MODEL
K=10
nLayers=5
training = True
out={}

# DEFINE THE SHARED LAYERS
Dn = mm.Denoiser(nLayers=nLayers,training=training) # nLayers >=3
Jaco  = mm.JAT(tes) 
Lambda={}
Mu={}
for k in range(1,K+1):
    Lambda[k] = mm.superPara(initial_value=0.05,trainable=True,name='Lambda'+str(k))
    Mu[k]     = mm.superPara(initial_value=0.05,trainable=True,name='Mu'+str(k))

# OUTPUT
Dc={}

# DEFINE THE MODEL 
Dc['dc0']=x0
for k in range(1,K+1):
    x=Dc['dc'+str(k-1)]
    Dx=Dn(x)
    Jr=Jaco([x,b])
    l1=tf.keras.layers.Subtract()([x,Dx])
    l2=tf.keras.layers.Add()([Jr,Lambda[k](l1)])
    l3=tf.keras.layers.Subtract()([x,2*Mu[k](l2)])
    Dc['dc'+str(k)]=l3
    # l1=tf.keras.layers.Subtract()([x, 2*Mu[k](Lambda[k](x))])
    # l2=tf.keras.layers.Add()([l1, 2*Lambda[k](Mu[k](Dx))])
    # l3=tf.keras.layers.Subtract()([l2,2*Mu[k](Jr)])
    # Dc['dc'+str(k)]=l3
    print('='*100)
    print('Dx:'+str(k)),helper.checkNaN(Dx)
    print('l1:'+str(k)),helper.checkNaN(l1)
    print('l2:'+str(k)),helper.checkNaN(l2)
    print('Jr:'+str(k)),helper.checkNaN(Jr)
    print('l3:'+str(k)),helper.checkNaN(l3)

outputs=Dc['dc'+str(K)]
o=outputs-x0
print('='*100)

print('o:'),helper.checkNaN(o)
print('x0:'),helper.checkNaN(x0)
print('outputs:'),helper.checkNaN(outputs)

print('dc1:'),helper.checkNaN(Dc['dc1'])
print('dc2:'),helper.checkNaN(Dc['dc2'])


