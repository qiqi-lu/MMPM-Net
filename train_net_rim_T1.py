import tensorflow as tf
import config
import functions as func
import functions_data as dfunc
import unrollingnet.RIM_T1 as rim
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
#######################################################################################
#### Model configuration
print('Create model ...')
Nc = len(id_tau) # number of weighted images
Ns = 6  # num of stage
f  = 36 # number of filters
Np = 3  # number of parameter maps
signed = 1 # signed/nosigned manitude data
share_weight = 0

nn       = rim.RIM(Ns=Ns,Np=Np,filters=f,signed=signed,share_weight=share_weight,test_mode=False,name='rim')
inpt_b   = tf.keras.layers.Input(shape=(None,None,Nc),name='images')
inpt_tau = tf.keras.layers.Input(shape=(Nc,),name='tes')
para_map = nn([inpt_b,inpt_tau]) 
model    = tf.keras.Model(inputs=[inpt_b,inpt_tau],outputs=para_map, name='RIM')

model.summary()

## TRAINING
#######################################################################################
#### Training configuration
batch_size = 10
epochs     = 300
save_every = 10
validation_batch_size = 10
ini_lr = 0.001

steps_per_epoch  = tf.math.floor(dataset_train_size/batch_size)
validation_steps = tf.math.floor(dataset_valid_size/validation_batch_size)

suffix    = 'maemow_lr_{}_bs_{}_'.format(ini_lr,batch_size)
model_dir = os.path.join('model',type_task,'rim','rim_Ns_{}_sw_{}_signed_{}_sigma_{}_{}'.format(Ns,share_weight,signed,sigma,suffix))
print(model_dir)
print('-'*98)

initial_epoch = func.findLastCheckpoint(save_dir=model_dir)
if initial_epoch > 0:
    print('Resuming by loading epoch %03d'%initial_epoch)
    model.load_weights(filepath=os.path.join(model_dir,'model_%03d.h5'%initial_epoch))

#######################################################################################
opti    = tf.keras.optimizers.Adam(learning_rate=ini_lr)
# loss    = tf.keras.losses.MeanSquaredError()
loss    = func.MAE_mow()
# loss    = func.l2norm()
metric  = [func.NRMSE_metric]

model.compile(optimizer=opti,loss=loss,metrics=metric)

checkpointer = tf.keras.callbacks.ModelCheckpoint(os.path.join(model_dir,'model_{epoch:03d}.h5'), verbose=1, save_weights_only=True, period=save_every)
csv_logger   = tf.keras.callbacks.CSVLogger(os.path.join(model_dir,'log.csv'), append=True, separator=',')
tensorboard  = tf.keras.callbacks.TensorBoard(log_dir=model_dir, histogram_freq=1)
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(func.lr_schedule)

model.fit(x=dataset_train.repeat(epochs).batch(batch_size,drop_remainder=True),
        epochs           = epochs,
        steps_per_epoch  = steps_per_epoch,
        validation_data  = dataset_valid.batch(validation_batch_size,drop_remainder=True),
        validation_steps = validation_steps,
        initial_epoch    = initial_epoch,
        callbacks        = [checkpointer,tensorboard,lr_scheduler],
        # callbacks        = [checkpointer,tensorboard],
        )