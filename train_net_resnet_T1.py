# ResNet for R1 mapping (training)

import os
import tensorflow as tf
import config
import functions as func
import CNN.ResNet as resnet

config.config_gpu(5)

#######################################################################################
id_tau = [0,1,2,3,4,5,6,7] # use all tau 
sigma = 'mix'
scale_factor = 400.0
type_task = 'T1mapping'

## DATASET
######################################################################################
#### Training datasets
print('='*98)
print('===== ',type_task,' =====')
print('Load dataset ...')
dataset_filenames = tf.io.gfile.glob(os.path.join('data',type_task,'trainingdata','*45_te_0_sigma_{}*.tfrecords'.format(sigma)))
dataset = tf.data.TFRecordDataset(dataset_filenames)\
            .map(func.parse_all)\
            .map(lambda x1,x2,x3,x4,x5,x6,x7: func.extract(x1,x2,x3,x4,x5,x6,x7,\
                id_tau=id_tau,rescale=scale_factor,model_type='cnn',data_type='mono',task=type_task))

# split the data for training and validation
split = 0.8
dataset_size = func.get_len(dataset)
dataset_train_size = int(dataset_size*split)
dataset_valid_size = int(dataset_size*(1.0-split))

# validation data
dataset_valid = dataset.shard(num_shards=5,index=4)
# training data
dataset_train = dataset.shard(num_shards=5,index=0)
dataset_train = dataset_train.concatenate(dataset.shard(num_shards=5,index=1))
dataset_train = dataset_train.concatenate(dataset.shard(num_shards=5,index=2))
dataset_train = dataset_train.concatenate(dataset.shard(num_shards=5,index=3))

dataset_train.shuffle(1000,reshuffle_each_iteration=True,seed=8200)
dataset_valid.shuffle(1000,reshuffle_each_iteration=True,seed=8200)

print('-'*98)
print('Dataset')
print(dataset_filenames)
print('  Training data size : '+str(dataset_train_size))
print('Validation data size : '+str(dataset_valid_size))
print('-'*98)

## MODEL
#######################################################################################
#### Model configuration
print('Create model ...')
Nc = len(id_tau) # the number of channels
Nb = 4 # the number of residual block
Np = 3 # the number of output parameter maps

model = resnet.ResNet_moled(image_channels=Nc,output_channel=Np,num_block=Nb)
model.summary()

## TRAINING
#######################################################################################
#### Training configuration
batch_size = 64 # 10, 32, 64
validation_batch_size = 64
epochs     = 1000
save_every = 10
lr = 0.00025

steps_per_epoch  = tf.math.floor(dataset_train_size/batch_size)
validation_steps = tf.math.floor(dataset_valid_size/validation_batch_size)

suffix    = 'moled_sigma_{}_bs_{}_mae_lr_{}'.format(sigma,batch_size,lr)
model_dir = os.path.join('model',type_task,'resnet','resnet_Nb_{}_'.format(Nb)+suffix)
print(model_dir)

initial_epoch = func.findLastCheckpoint(save_dir=model_dir)
if initial_epoch > 0:
    print('Resuming by loading epoch %03d'%initial_epoch)
    model.load_weights(filepath=os.path.join(model_dir,'model_%03d.h5'%initial_epoch))

#######################################################################################
opti = tf.keras.optimizers.Adam(learning_rate=lr)

loss = func.MAE()
# loss = tf.keras.losses.MeanSquaredError()
metric = [func.NRMSE_metric_cnn]

model.compile(optimizer=opti,loss=loss,metrics=metric)

checkpointer = tf.keras.callbacks.ModelCheckpoint(os.path.join(model_dir,'model_{epoch:03d}.h5'), verbose=1, save_weights_only=True, period=save_every)
csv_logger   = tf.keras.callbacks.CSVLogger(os.path.join(model_dir,'log.csv'), append=True, separator=',')
tensorboard  = tf.keras.callbacks.TensorBoard(log_dir=model_dir, histogram_freq=1)

model.fit(x=dataset_train.repeat(epochs).batch(batch_size,drop_remainder=True),
        epochs           = epochs,
        steps_per_epoch  = steps_per_epoch,
        validation_data  = dataset_valid.batch(validation_batch_size,drop_remainder=True),
        validation_steps = validation_steps,
        initial_epoch    = initial_epoch,
        callbacks        = [checkpointer,tensorboard],
        )