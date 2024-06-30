"""
Custom MoDL model using Tensorflow.

Reference: 
MoDL: Model-Based Deep Learning Architecture for Inverse Problems
by H.K. Aggarwal, M.P. Mani, M. Jacob from University of Iowa.

THis code solves the following optimization problem:
argmin_X ||A(x)-b||_2^2 + ||x-Dw(x)||_2^2

"""
# import supporting package
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# import cunstom function
import config
import helper
import model as mod
import callbacks as cb

# SET GPU
gpu=1
config.config_gpu(gpu)

# SET PARAMETERS (training)
epochs=600
batchSize=256 # for LIVER
save_every=20

# SET PARAMETERS (model)
nLayers=5
K=10
training = True
tes = np.array([0.93, 2.27, 3.61, 4.95, 6.29, 7.63, 8.97, 10.4, 11.8, 13.2, 14.6, 16.0])

# DATA PARAMETERS
sg=10 # nosie standard deviation
noise_type='Gaussian'
# noise_type='Rician'
data_type='LIVER'
# data_type='PHANTOM'
pattern=1
# pattern=2
# pattern=3

data_dir  = os.path.join('data',data_type+str(pattern),noise_type,str(sg),'TRAIN')

# DIR CONFIGATION
# custom_label = 'DOPAMINEe-4'
# custom_label = 'DOPAMINE'
# custom_label = 'CNN'
custom_label = 'MoDL'
saveDir   = os.path.join('savedModels',data_type)
model_dir = os.path.join(saveDir,str(nLayers)+'L_'+str(K)+'K_'+noise_type+custom_label)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
log_dir   = os.path.join('logs','fit',data_type,str(nLayers)+'L_'+str(K)+'K_'+noise_type+custom_label)
fw = tf.summary.create_file_writer(os.path.join(log_dir))
fw.set_as_default()

# DATA
print('='*98)
print('Load data...')
imgs   = np.load(os.path.join(data_dir,'wPatch.npy'))
imgs_n = np.load(os.path.join(data_dir,'wPatchN.npy'))
maps   = np.load(os.path.join(data_dir,'pPatch.npy'))
masks  = np.load(os.path.join('data',data_type+str(pattern),'maskPatchTrn.npy'))

indices = list(range(imgs.shape[0]))
np.random.shuffle(indices)

imgs   = imgs[indices]
imgs_n = imgs_n[indices]
maps   = maps[indices]
masks  = masks[indices]

print('Checking data NaN...')
helper.checkNaN(imgs)
helper.checkNaN(imgs_n)
helper.checkNaN(maps)

split=int(imgs.shape[0]*0.8)

b_train  = imgs_n[0:split]
# x0_train = helper.LogLinear2(b_train,tes)
x0_train = helper.LogLinearN(b_train,tes,n=2)
m_train  = masks[0:split]
x0_train = x0_train*m_train
x0_train[x0_train>2000] = 100
x_train  = maps[0:split]

b_valid  = imgs_n[split:]
# x0_valid = helper.LogLinear2(b_valid,tes)
x0_valid = helper.LogLinearN(b_valid,tes,n=2)
m_valid  = masks[split:]
x0_valid = x0_valid*m_valid
x0_valid[x0_valid>2000] = 100
x_valid  = maps[split:]

print('Training data information:')
print('  b : ',b_train.shape)
print('  x0: ',x0_train.shape)
print('  m : ',m_train.shape)
print('  x : ',x_train.shape)

# Show training data sample
plt.figure(figsize=(20,10))
id=10
plt.subplot(3,4,1),plt.imshow(b_train[id,:,:,0],vmax=400,vmin=0,cmap='gray'),plt.title('$T_2$ weighted images',loc='left')
plt.subplot(3,4,2),plt.imshow(b_train[id,:,:,1],vmax=400,vmin=0,cmap='gray')
plt.subplot(3,4,3),plt.imshow(b_train[id,:,:,2],vmax=400,vmin=0,cmap='gray')
plt.subplot(3,4,4),plt.imshow(b_train[id,:,:,3],vmax=400,vmin=0,cmap='gray')
plt.subplot(3,4,5),plt.imshow(x0_train[id,:,:,0],vmax=400,vmin=0,cmap='jet'),plt.title('$x_0$',loc='left'),plt.title('$S_0$')
plt.subplot(3,4,6),plt.imshow(x0_train[id,:,:,1],vmax=1100,vmin=0,cmap='jet'),plt.title('$R_2$')
plt.subplot(3,4,7),plt.imshow(x_train[id,:,:,0],vmax=400,vmin=0,cmap='jet'),plt.title('$x$',loc='left'),plt.title('$S_0$')
plt.subplot(3,4,8),plt.imshow(x_train[id,:,:,1],vmax=1100,vmin=0,cmap='jet'),plt.title('$R_2$')
plt.subplot(3,4,9),plt.imshow(m_train[id,:,:,0],cmap='gray')
plt.subplot(3,4,10),plt.imshow(m_train[id,:,:,1],cmap='gray')
plt.savefig(os.path.join('figures','train_data'+str(sg)+'.png'))

print('='*100)
print('Creating model...')

TEs = tf.convert_to_tensor(tes,dtype=tf.float32)
# model=mm.MoDL(nLayers=nLayers,K=K,training=training,tes=TEs)
model=mod.MoDLFixMu(nLayers=nLayers,K=K,tes=TEs,BN=True)
# model=mm.DenoiserModel(nLayers=nLayers,K=K,training=training,tes=tes)
# model=mm.IterModel(nLayers=nLayers,K=K,training=training,tes=tes)
# model = mod.DenoiserModel(nLayers=nLayers,K=K,tes=TEs) # CNN
# model = mod.DOPAMINE(nLayers=nLayers,K=K,tes=TEs) # CNN

model.build([(None,None,None,2),(None,None,None,12)])
model.summary()

# load the last model weights
initial_epoch = helper.findLastCheckpoint(save_dir=model_dir)
if initial_epoch > 0:  
    print('Resuming by loading epoch %03d'%initial_epoch)
    model.load_weights(filepath=os.path.join(model_dir,'model_%03d.h5'%initial_epoch))


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.000005,beta_1=0.9,beta_2=0.999,epsilon=1e-07),
              loss=tf.keras.losses.MeanSquaredError()
            #   loss=tf.keras.losses.LogCosh()
            #   loss=tf.keras.losses.Huber(delta=100.0)
              # loss=tf.keras.losses.MeanAbsoluteError()
              )




checkpointer = tf.keras.callbacks.ModelCheckpoint(os.path.join(model_dir,'model_{epoch:03d}.h5'), 
            verbose=1, save_weights_only=True, period=save_every)
    
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
# lr_scheduler=tf.keras.callbacks.LearningRateScheduler(helper.lr_schedule)
# lr_scheduler=tf.keras.callbacks.LearningRateScheduler(helper.lr_schedule2)
# lr_scheduler=tf.keras.callbacks.LearningRateScheduler(helper.lr_schedule3)
# lr_scheduler=tf.keras.callbacks.LearningRateScheduler(helper.lr_schedule4)


print('='*98)
print('Training...')
# model.fit(x=[x0_train,b_train,m_train],
#           y=x_train,
#           epochs=epochs,
#           batch_size=batchSize,
#           validation_data=([x0_valid,b_valid,m_valid], x_valid),
#         #   shuffle=False, 
#           callbacks=[tensorboard_callback,checkpointer])
#           # callbacks=[tensorboard_callback,checkpointer,lr_scheduler])

logpara = cb.LogSuperPara(K=K,Lambda_type='Same')
model.fit(x=[x0_train,b_train],
          y=x_train,
          epochs=epochs,
          batch_size=batchSize,
          validation_data=([x0_valid,b_valid], x_valid),
        #   shuffle=False, 
          callbacks=[tensorboard_callback,checkpointer,logpara])
          # callbacks=[tensorboard_callback,checkpointer,lr_scheduler])

# model.fit(x=x0_train,
#           y=x_train,
#           epochs=epochs,
#           batch_size=batchSize,
#           validation_data=(x0_valid, x_valid), 
#           callbacks=[tensorboard_callback,checkpointer])
