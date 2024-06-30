import tensorflow as tf
import model_modl as mm
import numpy as np
import matplotlib.pyplot as plt
import config
import helper
import os
config.config_gpu(3)

# DATA
# Parameter maps (Reference)
size=(10,10) #block image size
size_block=(10,10)
n=10 # number of study
sg=30 # nosie standard deviation

x=np.zeros((n,size[0]*size_block[0],size[1]*size_block[1],2))
for i in range(n):
    x[...,0],_=helper.makeBlockImage(img_size=size,block_size=(10,10),type='Random',value=[300,400])
    x[...,1],_=helper.makeBlockImage(img_size=size,block_size=(10,10),type='Random',value=[10,1000])
print('Parameter maps (refernce) shape: ',x.shape)

# Make training data
tes = tf.constant([0.93, 2.27, 3.61, 4.95, 6.29, 7.63, 8.97, 10.4, 11.8, 13.2, 14.6, 16.0])

imgs,imgs_n,maps=helper.makePairedData(para_maps=x,tes=tes,sigma=sg,noise_type='Rician')

b =imgs_n
x0=helper.LogLinear2(b,tes)
x =maps

# Show training data sample
plt.figure(figsize=(20,10))
id=1
plt.subplot(2,4,1),plt.imshow(b[id,:,:,0],vmax=450,vmin=0,cmap='gray'),plt.title('TE(1)'),plt.colorbar(fraction=0.022)
plt.subplot(2,4,2),plt.imshow(b[id,:,:,1],vmax=450,vmin=0,cmap='gray'),plt.title('TE(2)'),plt.colorbar(fraction=0.022)
plt.subplot(2,4,3),plt.imshow(b[id,:,:,2],vmax=450,vmin=0,cmap='gray'),plt.title('TE(3)'),plt.colorbar(fraction=0.022)
plt.subplot(2,4,4),plt.imshow(b[id,:,:,3],vmax=450,vmin=0,cmap='gray'),plt.title('TE(4)'),plt.colorbar(fraction=0.022)
plt.subplot(2,4,5),plt.imshow(x0[id,:,:,0],vmax=450,vmin=250,cmap='jet'),plt.title('S0(x0)'),plt.colorbar(fraction=0.022)
plt.subplot(2,4,6),plt.imshow(x0[id,:,:,1],vmax=1100,vmin=0,cmap='jet'),plt.title('R2*(x0)'),plt.colorbar(fraction=0.022)
plt.subplot(2,4,7),plt.imshow(x[id,:,:,0],vmax=450,vmin=250,cmap='jet'),plt.title('S0(x)'),plt.colorbar(fraction=0.022)
plt.subplot(2,4,8),plt.imshow(x[id,:,:,1],vmax=1100,vmin=0,cmap='jet'),plt.title('R2*(x)'),plt.colorbar(fraction=0.022)
plt.savefig(os.path.join('figures','train_data'+str(sg)+'.png'))

# MODEL
JAT=mm.JAT(tes)
A=mm.A(tes)

Ax=A(x0)
Axb=Ax-b
plt.figure(figsize=(20,10))
id=1
plt.subplot(3,4,1),plt.imshow(b[id,:,:,0],vmax=450,vmin=0,cmap='gray'),plt.title('TE(1)'),plt.colorbar(fraction=0.022),plt.title('b',loc='left')
plt.subplot(3,4,2),plt.imshow(b[id,:,:,1],vmax=450,vmin=0,cmap='gray'),plt.title('TE(2)'),plt.colorbar(fraction=0.022)
plt.subplot(3,4,3),plt.imshow(b[id,:,:,2],vmax=450,vmin=0,cmap='gray'),plt.title('TE(3)'),plt.colorbar(fraction=0.022)
plt.subplot(3,4,4),plt.imshow(b[id,:,:,3],vmax=450,vmin=0,cmap='gray'),plt.title('TE(4)'),plt.colorbar(fraction=0.022)
plt.subplot(3,4,5),plt.imshow(Ax[id,:,:,0],vmax=450,vmin=0,cmap='gray'),plt.title('TE(1)'),plt.colorbar(fraction=0.022),plt.title('Ax',loc='left')
plt.subplot(3,4,6),plt.imshow(Ax[id,:,:,1],vmax=450,vmin=0,cmap='gray'),plt.title('TE(2)'),plt.colorbar(fraction=0.022)
plt.subplot(3,4,7),plt.imshow(Ax[id,:,:,2],vmax=450,vmin=0,cmap='gray'),plt.title('TE(3)'),plt.colorbar(fraction=0.022)
plt.subplot(3,4,8),plt.imshow(Ax[id,:,:,3],vmax=450,vmin=0,cmap='gray'),plt.title('TE(4)'),plt.colorbar(fraction=0.022)
plt.subplot(3,4,9),plt.imshow(Axb[id,:,:,0],vmax=100,vmin=-100,cmap='gray'),plt.title('TE(1)'),plt.colorbar(fraction=0.022),plt.title('Axb',loc='left')
plt.subplot(3,4,10),plt.imshow(Axb[id,:,:,1],vmax=100,vmin=-100,cmap='gray'),plt.title('TE(2)'),plt.colorbar(fraction=0.022)
plt.subplot(3,4,11),plt.imshow(Axb[id,:,:,2],vmax=100,vmin=-100,cmap='gray'),plt.title('TE(3)'),plt.colorbar(fraction=0.022)
plt.subplot(3,4,12),plt.imshow(Axb[id,:,:,3],vmax=100,vmin=-100,cmap='gray'),plt.title('TE(4)'),plt.colorbar(fraction=0.022)
plt.savefig(os.path.join('figures','Ax'+str(sg)+'.png'))


Jr=JAT([x0,b])
# mu=1
mu=0.1
x1=x0-2*mu*Jr
plt.figure(figsize=(20,10))
id=1
plt.subplot(2,4,1),plt.imshow(x0[id,:,:,0],vmax=450,vmin=0,cmap='jet'),plt.title('s0'),plt.colorbar(fraction=0.022),plt.title('x0',loc='left')
plt.subplot(2,4,5),plt.imshow(x0[id,:,:,1],vmax=1100,vmin=0,cmap='jet'),plt.title('R2*'),plt.colorbar(fraction=0.022)
plt.subplot(2,4,2),plt.imshow(x1[id,:,:,0],vmax=450,vmin=0,cmap='jet'),plt.title('s0'),plt.colorbar(fraction=0.022),plt.title('x1',loc='left')
plt.subplot(2,4,6),plt.imshow(x1[id,:,:,1],vmax=1100,vmin=0,cmap='jet'),plt.title('R2*'),plt.colorbar(fraction=0.022)
plt.subplot(2,4,3),plt.imshow(x[id,:,:,0],vmax=450,vmin=0,cmap='jet'),plt.title('s0'),plt.colorbar(fraction=0.022),plt.title('x',loc='left')
plt.subplot(2,4,7),plt.imshow(x[id,:,:,1],vmax=1100,vmin=0,cmap='jet'),plt.title('R2*'),plt.colorbar(fraction=0.022)
plt.subplot(2,4,4),plt.imshow(Jr[id,:,:,0],vmax=450,vmin=-450,cmap='jet'),plt.title('s0'),plt.colorbar(fraction=0.022),plt.title('Jr',loc='left')
plt.subplot(2,4,8),plt.imshow(Jr[id,:,:,1],vmax=1100,vmin=-1100,cmap='jet'),plt.title('R2*'),plt.colorbar(fraction=0.022)

plt.savefig(os.path.join('figures','JAT'+str(sg)+'.png'))

