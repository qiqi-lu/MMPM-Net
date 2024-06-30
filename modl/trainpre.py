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
gpu=7
config.config_gpu(gpu)

# TRAINING PARAMETERS
epochs=600
# batchSize=260 # for LIVER
batchSize=32 # for LIVER
save_every=20

# MODEL PARAMETERS
tes = np.array([0.93, 2.27, 3.61, 4.95, 6.29, 7.63, 8.97, 10.4, 11.8, 13.2, 14.6, 16.0])
nLayers=5
K  = 10

# Opitimization Strategy
OS = 'GD' 
# OS = 'CG'

# Training Strategy
TS = 'ET' # End-to-end Training
# TS = 'PD'

# Network Architecture
NA = 'WS'
# NA = 'NS'

# Lambda Type
# lambda_type = 'Diff'
lambda_type = 'Same'
# lambda_type = 10

# Separate = True
Separate = False
No=1

# init = 0 
init = 3 # LogLinear method input number

maskOut = True
# maskOut = False

# DATA PARAMETERS
# sg = 10 # nosie standard deviation
sg = 20

noise_type = 'Gaussian'
# noise_type='Rician'

data_type = 'LIVER'
# data_type='PHANTOM'
pattern=1
# pattern=2
# pattern=3

data_dir  = os.path.join('data',data_type+str(pattern),noise_type,str(sg),'TRAIN')

# name = 'DOPAMINE'
# name = 'Denoiser'
name = 'MoDL'

saveDir    = os.path.join('savedModels',data_type+str(pattern))
model_name = name+'_'+OS+'_'+TS+'_'+NA+'_'+lambda_type+'_'+str(nLayers)+'L_'+str(K)+'K_'+str(sg)+noise_type
model_dir  = os.path.join(saveDir,model_name)
log_dir    = os.path.join('logs',data_type+str(pattern),model_name)

if not os.path.exists(model_dir): os.makedirs(model_dir)
fw = tf.summary.create_file_writer(os.path.join(log_dir))
fw.set_as_default()

# DATA
print('='*98+'\nLoad data...')
# imgs   = np.load(os.path.join(data_dir,'wPatch.npy'))
b = np.load(os.path.join(data_dir,'wPatchN.npy'))
x = np.load(os.path.join(data_dir,'pPatch.npy'))
m = np.load(os.path.join('data',data_type+str(pattern),'maskPatchTrn.npy'))

indices = list(range(b.shape[0]))
np.random.shuffle(indices)

# imgs   = imgs[indices]
b = b[indices]
x = x[indices]
m = m[indices]
if init==0: x0=np.zeros(x.shape)
if init>1 : x0= helper.LogLinearN(b,tes,n=init)
if maskOut==True: x0=x0*m
x0[x0>2000] = 100

print('Checking data NaN...')
helper.checkNaN(b)
helper.checkNaN(x)

# split into training and validation data
split=int(b.shape[0]*0.8)

m_train  = m[0:split]
b_train  = b[0:split]
x_train  = x[0:split]
x0_train = x0[0:split]

m_valid  = x[split:]
b_valid  = b[split:]
x_valid  = x[split:]
x0_valid = x0[split:]

print('Training data information:'+'\n- b : '+str(b_train.shape)+'\n- x0: '+str(x0_train.shape)+'\n- m : '+str(m_train.shape)+'\n- x : '+str(x_train.shape))

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

print('='*98+'\nCreating model...')
TEs = tf.convert_to_tensor(tes,dtype=tf.float32)

if name=='MoDL':model = mod.MoDL(nLayers=nLayers,K=K,tes=TEs,Lambda_type=lambda_type,BN=True,OS=OS,TS=TS,NA=NA)
if name=='Denoiser':model = mod.Denoiser(nLayers=nLayers,Seperated=Separate,NA=NA,n=No)
if name=='DOPAMINE': model = mod.DOPAMINE(nLayers=nLayers,K=K,tes=TEs,TS=TS)

# model.build([(None,None,None,2),(None,None,None,12)])
model.build((None,None,None,2))
model.summary()

# load the last model weights
init_epoch = helper.findLastCheckpoint(save_dir=model_dir)
if init_epoch > 0:
    print('Resuming by loading epoch %03d'%init_epoch)
    model.load_weights(filepath=os.path.join(model_dir,'model_%03d.h5'%init_epoch))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001,beta_1=0.9,beta_2=0.999,epsilon=1e-07),
              loss=tf.keras.losses.MeanSquaredError()
              # loss=tf.keras.losses.LogCosh()
              # loss=tf.keras.losses.Huber(delta=100.0)
              # loss=tf.keras.losses.MeanAbsoluteError()
              )

# callbacks
checkpointer = tf.keras.callbacks.ModelCheckpoint(os.path.join(model_dir,'model_{epoch:03d}.h5'),verbose=1, save_weights_only=True, period=save_every)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
logpara = cb.LogSuperPara(K=K,Lambda_type=lambda_type)
# lr_scheduler=tf.keras.callbacks.LearningRateScheduler(cb.lr_schedule)

print('='*98+'\nTraining...')

# model.fit(x=[x0_train,b_train],
#           y=x_train,
#           epochs=epochs,
#           batch_size=batchSize,
#           validation_data=([x0_valid,b_valid], x_valid),
#           callbacks=[tensorboard_callback,checkpointer,logpara])
#           # callbacks=[tensorboard_callback,checkpointer,lr_scheduler])

model.fit(x=x0_train,
          y=x_train,
          epochs=epochs,
          batch_size=batchSize,
          validation_data=(x0_valid, x_valid), 
          callbacks=[tensorboard_callback,checkpointer])
            # callbacks=[tensorboard_callback,checkpointer,lr_scheduler])

