import tensorflow as tf
import os
import functions as func

def load_training_data(file_name,id_tau,rescale_para,type_task,):
    dataset_filenames = tf.io.gfile.glob(file_name)
    dataset = tf.data.TFRecordDataset(dataset_filenames)\
                .map(func.parse_all)\
                .map(lambda x1,x2,x3,x4,x5,x6,x7: func.extract(x1,x2,x3,x4,x5,x6,x7,\
                    id_tau=id_tau,rescale=rescale_para,model_type='unrolling',data_type='mono',task=type_task))


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

    print('Reshuffle ...')
    dataset_train.shuffle(1000,reshuffle_each_iteration=True)
    dataset_valid.shuffle(1000,reshuffle_each_iteration=True)

    print('-'*98)
    print('Dataset')
    print(dataset_filenames)                                                                                                                                                       
    print('  Training data size : {}'.format(dataset_train_size))
    print('Validation data size : {}'.format(dataset_valid_size))
    for sample in dataset.take(1): 
        tau = sample[0][1]
        print(sample[0][0].shape)
    print('TE: '+str(tau))
    print('-'*98)
    return dataset_train, dataset_valid, dataset_train_size, dataset_valid_size
