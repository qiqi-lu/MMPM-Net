import tensorflow as tf
import model as mod
import os
import matplotlib.pyplot as plt
import helper
import numpy as np
import config

# SET GPU
gpu=1
config.config_gpu(gpu)

# SET PARAMETERS (model)
nLayers=5

# K=10
K=1

tes = tf.constant([0.93, 2.27, 3.61, 4.95, 6.29, 7.63, 8.97, 10.4, 11.8, 13.2, 14.6, 16.0])
# tes = tf.constant([0.99, 2.40, 3.81, 5.22, 6.63, 8.04, 9.45, 10.86,12.27,13.68,15.09,16.5])
# tes = tf.constant([0.80, 1.05, 1.30, 1.55, 1.80, 2.05, 2.30, 2.55, 2.80, 3.05, 3.30, 3.55])

# model=mod.MoDL(nLayers=nLayers,K=K,tes=tes)
# model=mod.MoDLFixMu(nLayers=nLayers,K=K,tes=tes)
# model=mod.MoDLFixMuMask(nLayers=nLayers,K=K,tes=tes)
model=mod.DenoiserModel(nLayers=nLayers,K=K,tes=tes)
# model=mod.DOPAMINE(nLayers=nLayers,K=K,tes=tes)

# DATA
# data_type ='PHANTOM'
data_type ='LIVER'
pattern=1
# pattern=2
# pattern=3

# noise_type='Rician'
noise_type='Gaussian'
sg=10

print('='*100)
print('Load Saved Data (Testing)...')
data_dir  = os.path.join('data',data_type+str(pattern),noise_type,str(sg),'TEST')
print(data_dir)
imgs   = np.load(os.path.join(data_dir,'wImg.npy'))
imgs_n = np.load(os.path.join(data_dir,'wImgN.npy'))
maps   = np.load(os.path.join(data_dir,'pImg.npy'))
masks_body  = np.load(os.path.join('data',data_type+str(pattern),'maskBody.npy'))[100:]
masks_liver  = np.load(os.path.join('data',data_type+str(pattern),'maskLiver.npy'))[100:]
# masks  = np.load(os.path.join('data',data_type+str(pattern),'maskParenchyma.npy'))[100:]

x_test_b  = imgs_n
# x_test_x0 = helper.LogLinear2(x_test_b,tes)
x_test_x0 = helper.LogLinearN(x_test_b,tes,n=2)
x_test_m  = masks_body
y_test_x  = maps

print('Test data information:')
print(' b: ', x_test_b.shape)
print(' x0:',x_test_x0.shape)
print(' M: ', x_test_m.shape)
print(' x: ', y_test_x.shape)

print('='*100)
model_id='600'
model_type='LIVER'
# model_type='PHANTOM'
custom_label='CNN'
# custom_label='DOPAMINE'
# custom_label='DOPAMINEe-4'
model_dir=os.path.join('savedModels',model_type,str(nLayers)+'L_'+str(K)+'K_'+noise_type+custom_label,'model_'+model_id+'.h5')

print('Load model '+model_dir)
model.load_weights(model_dir)

# pred=model.predict([x_test_x0,x_test_b,x_test_m])
pred=model.predict([x_test_x0*x_test_m,x_test_b]) # DOPAMINE
# pred=model.predict(x_test_x0) #CNN

m  = masks_liver
x  = y_test_x*x_test_m
x0 = x_test_x0*x_test_m
p  = pred*x_test_m

import metricx
plt.figure(figsize=(25,25))
i=0

NRMSE0s = metricx.nRMSE(x[...,0],x0[...,0],m)
NRMSE0r = metricx.nRMSE(x[...,1],x0[...,1],m)
SSIM0s  = metricx.SSIM(x[...,0],x0[...,0],m)
SSIM0r  = metricx.SSIM(x[...,1],x0[...,1],m)

NRMSEps = metricx.nRMSE(x[...,0],p[...,0],m)
NRMSEpr = metricx.nRMSE(x[...,1],p[...,1],m)
SSIMps  = metricx.SSIM(x[...,0],p[...,0],m)
SSIMpr  = metricx.SSIM(x[...,1],p[...,1],m)

plt.subplot(4,3,1),plt.imshow(x[i,:,:,0],cmap='jet',vmax=500,vmin=0,interpolation='none'),plt.title('x'),plt.colorbar(fraction=0.025),plt.title('$S_0$',loc='left')
plt.subplot(4,3,2),plt.imshow(x0[i,:,:,0],cmap='jet',vmax=500,vmin=0,interpolation='none'),plt.title('$x_0$'),plt.colorbar(fraction=0.025)
plt.subplot(4,3,3),plt.imshow(p[i,:,:,0],cmap='jet',vmax=500,vmin=0,interpolation='none'),plt.title('$x_p$'),plt.colorbar(fraction=0.025)
plt.subplot(4,3,5),plt.imshow(abs(x0[i,:,:,0]-x[i,:,:,0]),cmap='jet',vmax=100,vmin=0,interpolation='none')
plt.colorbar(fraction=0.025),plt.title('$x_0$ vs x'),plt.title('nRMSE='+str(NRMSE0s[i]),loc='left'),plt.title('SSIM='+str(SSIM0s[i]),loc='right')
plt.subplot(4,3,6),plt.imshow(abs(p[i,:,:,0]-x[i,:,:,0]),cmap='jet',vmax=100,vmin=0,interpolation='none')
plt.colorbar(fraction=0.025),plt.title('$x_p$ vs x'),plt.title('nRMSE='+str(NRMSEps[i]),loc='left'),plt.title('SSIM='+str(SSIMps[i]),loc='right')

plt.subplot(4,3,7),plt.imshow(x[i,:,:,1],cmap='jet',vmax=800,vmin=0,interpolation='none'),plt.colorbar(fraction=0.025),plt.title('x'),plt.title('$R_2$',loc='left')
plt.subplot(4,3,8),plt.imshow(x0[i,:,:,1],cmap='jet',vmax=800,vmin=0,interpolation='none'),plt.colorbar(fraction=0.025),plt.title('$x_0$')
plt.subplot(4,3,9),plt.imshow(p[i,:,:,1],cmap='jet',vmax=800,vmin=0,interpolation='none'),plt.colorbar(fraction=0.025),plt.title('$x_p$')
plt.subplot(4,3,11),plt.imshow(abs(x0[i,:,:,1]-x[i,:,:,1]),cmap='jet',vmax=100,vmin=0,interpolation='none')
plt.colorbar(fraction=0.025),plt.title('$x_0$ vs x'),plt.title('nRMSE='+str(NRMSE0r[i]),loc='left'),plt.title('SSIM='+str(SSIM0r[i]),loc='right')
plt.subplot(4,3,12),plt.imshow(abs(p[i,:,:,1]-x[i,:,:,1]),cmap='jet',vmax=100,vmin=0,interpolation='none')
plt.colorbar(fraction=0.025),plt.title('$x_p$ vs x'),plt.title('nRMSE='+str(NRMSEpr[i]),loc='left'),plt.title('SSIM='+str(SSIMpr[i]),loc='right')
plt.savefig(os.path.join('figures','prediction.png'))


print('Data shape: ',x_test_x0.shape)
print('nRMSE(Prediction): ',[np.mean(NRMSEps),np.mean(NRMSEpr)])
print('nRMSE(x0):         ',[np.mean(NRMSE0s),np.mean(NRMSE0r)])
print('SSIM (Prediction): ',[np.mean(SSIMps),np.mean(SSIMpr)])
print('SSIM (x0):         ',[np.mean(SSIM0s),np.mean(SSIM0r)])

print(NRMSEpr)
print(SSIMpr)