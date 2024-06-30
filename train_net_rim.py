import tensorflow as tf
import config
import functions as func
import unrollingnet.RIM as rim
import os

config.config_gpu(5)

#######################################################################################
id_te = [0,1,3,7,15,31]
sigma = 'mix'

## DATASET
#######################################################################################
#### Training datasets
print('='*98)
print('Load data ...')
dataset_filenames = tf.io.gfile.glob(os.path.join('data','trainingdata','*45_te_1_sigma_{}*.tfrecords'.format(sigma)))
dataset = tf.data.TFRecordDataset(dataset_filenames)\
            .map(func.parse_all)\
            .map(lambda x1,x2,x3,x4,x5,x6,x7: func.extract(x1,x2,x3,x4,x5,x6,x7,\
                id_te=id_te,rescale=100.0,model_type='unrolling',data_type='mono'))
dataset_size      = func.get_len(dataset)

split = 0.8
dataset_train_size = int(dataset_size*split)
dataset_valid_size = int(dataset_size*(1.0-split))

print('Reshuffle ...')
dataset_valid = dataset.shard(num_shards=5,index=4)

dataset_train = dataset.shard(num_shards=5,index=0)
dataset_train = dataset_train.concatenate(dataset.shard(num_shards=5,index=1))
dataset_train = dataset_train.concatenate(dataset.shard(num_shards=5,index=2))
dataset_train = dataset_train.concatenate(dataset.shard(num_shards=5,index=3))

dataset_train.shuffle(1000,reshuffle_each_iteration=True)
dataset_valid.shuffle(1000,reshuffle_each_iteration=True)

print('-'*98)
print('Dataset')
print(dataset_filenames)
print('  Training data size : '+str(dataset_train_size))
print('Validation data size : '+str(dataset_valid_size))
for sample in dataset.take(1): te = sample[0][1]
print('TE: '+str(te))
print('-'*98)

## MODEL
#######################################################################################
#### Model configuration
print('Create model ...')
Nc = len(id_te)
Ns = 6 # num of stage
f  = 36
suffix = 'r2_mmse_mean_{}_{}_maemow_lr_10'.format(Nc,sigma)

nn       = rim.RIMm(Ns=Ns,filters=f,Np=2)
inpt_b   = tf.keras.layers.Input(shape=(None,None,Nc),name='images')
inpt_te  = tf.keras.layers.Input(shape=(Nc,),name='tes')
para_map = nn([inpt_b,inpt_te]) 
model    = tf.keras.Model(inputs=[inpt_b,inpt_te],outputs=para_map, name='RIM')

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

model_dir     = os.path.join('model','rim','rim_Ns_'+str(Ns)+'_f_'+str(f)+'_'+suffix)
initial_epoch = func.findLastCheckpoint(save_dir=model_dir)
if initial_epoch > 0:
    print('Resuming by loading epoch %03d'%initial_epoch)
    model.load_weights(filepath=os.path.join(model_dir,'model_%03d.h5'%initial_epoch))

opti    = tf.keras.optimizers.Adam(learning_rate=0.001)
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