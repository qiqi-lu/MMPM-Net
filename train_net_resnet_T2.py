import os
import tensorflow as tf
import config
import functions as func
import CNN.ResNet as resnet

config.config_gpu(6)

#######################################################################################
id_te = [0,1,3,7,15,31]
sigma = 'mix'

## DATASET
######################################################################################
#### Training datasets
print('='*98)
print('Load dataset ...')
dataset_filenames = tf.io.gfile.glob(os.path.join('data','trainingdata','*45_te_1_sigma_{}*.tfrecords'.format(sigma)))
dataset = tf.data.TFRecordDataset(dataset_filenames)\
            .map(func.parse_all)\
            .map(lambda x1,x2,x3,x4,x5,x6,x7: func.extract(x1,x2,x3,x4,x5,x6,x7,\
                id_te=id_te,rescale=100.0,model_type='cnn',data_type='mono'))
dataset_size = func.get_len(dataset)

split = 0.8
dataset_train_size = int(dataset_size*split)
dataset_valid_size = int(dataset_size*(1.0-split))

dataset_valid = dataset.shard(num_shards=5,index=4)

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
# for sample in dataset.take(1): te = sample[0][1]
# print('TE: '+str(te))
print('-'*98)

## MODEL
#######################################################################################
#### Model configuration
print('Create model ...')
Nc = len(id_te)
Nb = 4
suffix = 'moled_r2_mae_mono_{}_{}_10'.format(Nc,sigma)

model   = resnet.ResNet_moled(image_channels=Nc,output_channel=2,num_block=Nb)
model.summary()

## TRAINING
#######################################################################################
#### Training configuration
batch_size = 10
epochs     = 300
save_every = 10

validation_batch_size = 10
steps_per_epoch       = tf.math.floor(dataset_train_size/batch_size)
validation_steps      = tf.math.floor(dataset_valid_size/validation_batch_size)

model_dir     = os.path.join('model','resnet','resnet_'+str(Nb)+'_'+suffix)
initial_epoch = func.findLastCheckpoint(save_dir=model_dir)
if initial_epoch > 0:
    print('Resuming by loading epoch %03d'%initial_epoch)
    model.load_weights(filepath=os.path.join(model_dir,'model_%03d.h5'%initial_epoch))

#######################################################################################
opti = tf.keras.optimizers.Adam(learning_rate=0.001)

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