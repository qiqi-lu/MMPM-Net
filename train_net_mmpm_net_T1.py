import os
import tensorflow as tf
import config
import functions as func
import functions_data as dfunc
import unrollingnet.MMPM_Net_T1 as net

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
#######################################################################################
#### Model configuration
print('Create model ...')
Nc = len(id_tau)
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

## TRAINING
#######################################################################################
#### Training configuration
batch_size = 20
epochs     = 300
save_every = 10
validation_batch_size = batch_size
ini_lr = 0.001

steps_per_epoch  = tf.math.floor(dataset_train_size/batch_size)
validation_steps = tf.math.floor(dataset_valid_size/validation_batch_size)

suffix = 'lam_{}_rho_{}_bs_{}_maemow_lr_{}_xfzz'.format(ini_lam, ini_rho, batch_size, ini_lr)
model_dir = os.path.join('model',type_task,'mmpm_net','mmpm_net_Ns_{}_Nk_{}_Nt_{}_signed_{}_sigma_{}_{}'\
                                                        .format(Ns,Nk,Nt,signed,sigma,suffix))
print(model_dir)
print('-'*98)

#######################################################################################
initial_epoch = func.findLastCheckpoint(save_dir=model_dir)
if initial_epoch > 0:
    print('Resuming by loading epoch %03d'%initial_epoch)
    model.load_weights(filepath=os.path.join(model_dir,'model_%03d.h5'%initial_epoch))

#######################################################################################
opti    = tf.keras.optimizers.Adam(learning_rate=ini_lr)
# loss    = tf.keras.losses.MeanSquaredError()
loss    = func.MAE_mow()
# loss    = func.MAE()
metric  = [func.NRMSE_metric]

model.compile(optimizer=opti,loss=loss,metrics=metric)

checkpointer = tf.keras.callbacks.ModelCheckpoint(os.path.join(model_dir,'model_{epoch:03d}.h5'), verbose=1, save_weights_only=True, period=save_every)
csv_logger   = tf.keras.callbacks.CSVLogger(os.path.join(model_dir,'log.csv'), append=True, separator=',')
tensorboard  = tf.keras.callbacks.TensorBoard(log_dir=model_dir, histogram_freq=1)
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(func.lr_schedule)

model.fit(x = dataset_train.repeat(epochs).batch(batch_size,drop_remainder=True),
        epochs           = epochs, 
        steps_per_epoch  = steps_per_epoch,
        validation_data  = dataset_valid.batch(validation_batch_size,drop_remainder=True),
        validation_steps = validation_steps,
        initial_epoch    = initial_epoch,
        callbacks        = [checkpointer,tensorboard,lr_scheduler],
        # callbacks        = [checkpointer,tensorboard],
        )