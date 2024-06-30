import tensorflow as tf
import config
import functions as func
import functions_data as dfunc
import unrollingnet.DOPAMINE_T1 as dopamine
import os

config.config_gpu(5)

#######################################################################################
id_tau = [0,1,2,3,4,5,6,7]
sigma = 'mix'
rescale_para = 400.0
type_task = 'T1mapping'

file_name = os.path.join('data',type_task,'trainingdata','*45_te_0_sigma_{}*.tfrecords'.format(sigma))
## DATASET
#######################################################################################
#### Training datasets
print('='*98)
print('===== ',type_task,' =====')
print('Load data ...')
dataset_train, dataset_valid, dataset_train_size, dataset_valid_size =\
    dfunc.load_training_data(file_name=file_name,id_tau=id_tau,rescale_para=rescale_para,type_task=type_task)
   
## MODEL
##################################################################################
#### Model configuration
print('Create model ...')
Nc   = len(id_tau)  # number of weighted images
Ns   = 15           # number of stages
sep  = 0            # process each channel separately
norm = 0
signed = 1
share_weight = 0
ini_mu, ini_lm = 0.01, 0.01

nn = dopamine.DOPAMINE(Ns=Ns,sep=sep,norm=norm,signed=signed,share_weight=share_weight,test_mode=False,name='dopamine')
inpt_b  = tf.keras.layers.Input(shape=(None,None,Nc),name='images')
inpt_te = tf.keras.layers.Input(shape=(Nc,),name='tau')
para_map= nn([inpt_b,inpt_te])
model   = tf.keras.Model(inputs=[inpt_b,inpt_te],outputs=para_map)

model.summary()

## TRAINING
##################################################################################
#### Training configuration
batch_size = 10
epochs     = 300
save_every = 10
validation_batch_size = 10
ini_lr = 0.001

steps_per_epoch  = tf.math.floor(dataset_train_size/batch_size)
validation_steps = tf.math.floor(dataset_valid_size/validation_batch_size)

suffix    = 'maemow_lr_{}_bs_{}_mu_{}_lm_{}_dxx'.format(ini_lr,batch_size,ini_mu,ini_lm)
model_dir = os.path.join('model',type_task,'dopamine','dopamine_Ns_{}_norm_{}_sw_{}_sep_{}_signed_{}_sigma_{}_{}'\
            .format(Ns,norm,share_weight,sep,signed,sigma,suffix))
print(model_dir)
print('-'*98)

#######################################################################################
initial_epoch = func.findLastCheckpoint(save_dir=model_dir)
if initial_epoch > 0:
    print('Resuming by loading epoch {:03d}'.format(initial_epoch))
    model.load_weights(filepath=os.path.join(model_dir,'model_{:03d}.h5'.format(initial_epoch)))

#######################################################################################
opti   = tf.keras.optimizers.Adam(learning_rate=ini_lr)
# loss   = tf.keras.losses.MeanSquaredError()
loss   = func.MAE_mow()
# loss   = func.MAE()
metric = [func.NRMSE_metric]

model.compile(optimizer=opti,loss=loss,metrics=metric)

checkpointer = tf.keras.callbacks.ModelCheckpoint(os.path.join(model_dir,'model_{epoch:03d}.h5'), verbose=1, save_weights_only=True, period=save_every)
tensorboard  = tf.keras.callbacks.TensorBoard(log_dir=model_dir, histogram_freq=1)
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(func.lr_schedule)

model.fit(x=dataset_train.repeat(epochs).batch(batch_size,drop_remainder=True),
        epochs           = epochs,
        steps_per_epoch  = steps_per_epoch,
        validation_data  = dataset_valid.batch(validation_batch_size,drop_remainder=True),
        validation_steps = validation_steps,
        initial_epoch    = initial_epoch,
        # callbacks        = [checkpointer,tensorboard],
        callbacks        = [checkpointer,tensorboard,lr_scheduler]
        )