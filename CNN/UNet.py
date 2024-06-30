"""
UNets.
"""
import tensorflow as tf

def UNet(image_channels=12,output_channel=2):
    """
    Original U-Net architecture.
    [1] Ronneberger, O., Fischer, P. & Brox, T. U-net: Convolutional networks for biomedical image segmentation. (2015)
    """
    inpt  = tf.keras.layers.Input(shape=(None,None,image_channels)) # 12
    
    conv1 = tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same')(inpt)   # 64
    conv1 = tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same')(conv1)  # 64
    pool1 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(conv1) # 64
    
    conv2 = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(pool1) # 128
    conv2 = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(conv2) # 128
    pool2 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(conv2) # 128
    
    conv3 = tf.keras.layers.Conv2D(256,(3,3),activation='relu',padding='same')(pool2) # 256
    conv3 = tf.keras.layers.Conv2D(256,(3,3),activation='relu',padding='same')(conv3) # 256
    pool3 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(conv3) # 256
    
    conv4 = tf.keras.layers.Conv2D(512,(3,3),activation='relu',padding='same')(pool3) # 512
    conv4 = tf.keras.layers.Conv2D(512,(3,3),activation='relu',padding='same')(conv4) # 512
    pool4 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(conv4) # 512
    
    convbase = tf.keras.layers.Conv2D(1024,(3,3),activation='relu',padding='same')(pool4)    #1024
    convbase = tf.keras.layers.Conv2D(1024,(3,3),activation='relu',padding='same')(convbase) #1024
    
    # up-conv 2*2, an upsampling of the feature map followed by a 2*2 convolution ("up-convolution").
    conc5 = tf.keras.layers.Concatenate()([tf.keras.layers.Conv2D(512,(2,2),padding='same')(tf.keras.layers.UpSampling2D(size=(2,2))(convbase)),conv4]) #1024
    
    conv5 = tf.keras.layers.Conv2D(512,(3,3),activation='relu',padding='same')(conc5) # 512
    conv5 = tf.keras.layers.Conv2D(512,(3,3),activation='relu',padding='same')(conv5) # 512
    
    conc6 = tf.keras.layers.Concatenate()([tf.keras.layers.Conv2D(256,(2,2),padding='same')(tf.keras.layers.UpSampling2D(size=(2,2))(conv5)),conv3]) # 512
    
    conv6 = tf.keras.layers.Conv2D(256,(3,3),activation='relu',padding='same')(conc6) # 256
    conv6 = tf.keras.layers.Conv2D(256,(3,3),activation='relu',padding='same')(conv6) # 256
    
    conc7 = tf.keras.layers.Concatenate()([tf.keras.layers.Conv2D(128,(2,2),padding='same')(tf.keras.layers.UpSampling2D(size=(2,2))(conv6)),conv2]) # 256
    
    conv7 = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(conc7) # 128
    conv7 = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(conv7) # 128
    
    conc8 = tf.keras.layers.Concatenate()([tf.keras.layers.Conv2D(64, (2,2),padding='same')(tf.keras.layers.UpSampling2D(size=(2,2))(conv7)),conv1]) # 128
    
    conv8 = tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same')(conc8) # 64
    conv8 = tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same')(conv8) # 64
    
    convone = tf.keras.layers.Conv2D(output_channel,(1,1),activation='relu')(conv8) # 2
    
    model = tf.keras.Model(inputs=inpt,outputs=convone)
    return model

def UNet_residual(image_channels=12,output_channel=2):
    """
    Original U-Net architecture.
    [1] Liu, F., Kijowski, R., El Fakhri, G. & Feng, L. 
        Magnetic resonance parameter mapping using model‐guided self‐supervised deep learning. 
        Magn. Reson. Med. 85, 3211–3226 (2021).
    """
    inpt  = tf.keras.layers.Input(shape=(256,256,image_channels)) # 12
    
    conv1 = tf.keras.layers.Conv2D(filters=64,kernel_size=(4,4),stride=(2,2))(inpt) #128
    conv1 = tf.keras.layers.BatchNormalization()(conv1)

    conv2 = tf.keras.layers.Activation(tf.keras.activation.relu(alpha=0.2))(conv1)
    conv2 = tf.keras.layers.Conv2D(filters=128,kernel_size=(4,4),stride=(2,2))(conv1) #64
    conv2 = tf.keras.layers.BatchNormalization()(conv2)

    conv3 = tf.keras.layers.Activation(tf.keras.activation.relu(alpha=0.2))(conv2)
    conv3 = tf.keras.layers.Conv2D(filters=256,kernel_size=(4,4),stride=(2,2))(conv3) #32
    conv3 = tf.keras.layers.BatchNormalization()(conv3)

    conv4 = tf.keras.layers.Activation(tf.keras.activation.relu(alpha=0.2))(conv3)
    conv4 = tf.keras.layers.Conv2D(filters=512,kernel_size=(4,4),stride=(2,2))(conv4) #16
    conv4 = tf.keras.layers.BatchNormalization()(conv4)

    conv5 = tf.keras.layers.Activation(tf.keras.activation.relu(alpha=0.2))(conv4)
    conv5 = tf.keras.layers.Conv2D(filters=512,kernel_size=(4,4),stride=(2,2))(conv5) #8
    conv5 = tf.keras.layers.BatchNormalization()(conv5)

    conv6 = tf.keras.layers.Activation(tf.keras.activation.relu(alpha=0.2))(conv5)
    conv6 = tf.keras.layers.Conv2D(filters=512,kernel_size=(4,4),stride=(2,2))(conv6) #4
    conv6 = tf.keras.layers.BatchNormalization()(conv6)

    conv7 = tf.keras.layers.Activation(tf.keras.activation.relu(alpha=0.2))(conv6)
    conv7 = tf.keras.layers.Conv2D(filters=512,kernel_size=(4,4),stride=(2,2))(conv7) #2
    conv7 = tf.keras.layers.BatchNormalization()(conv7)

    base = tf.keras.layers.Activation(tf.keras.activation.relu(alpha=0.2))(conv7)
    base = tf.keras.layers.Conv2D(filters=512,kernel_size=(4,4),stride=(2,2))(base) #1
    base = tf.keras.layers.BatchNormalization()(base)
    
    deco7 = tf.keras.layers.Activation(tf.keras.activation.relu())(base)
    deco7 = tf.keras.layers.Conv2DTranspose(filters=512,kernel_size=(4,4),stride=(2,2))(deco7) #2
    deco7 = tf.keras.layers.BatchNormalization()(deco7)

    deco6 = tf.keras.layers.Concatenate()([conv7,deco7])
    deco6 = tf.keras.layers.Activation(tf.keras.activation.relu())(deco6)
    deco6 = tf.keras.layers.Conv2DTranspose(filters=512,kernel_size=(4,4),stride=(2,2))(deco6) #4
    deco6 = tf.keras.layers.BatchNormalization()(deco6)

    deco5 = tf.keras.layers.Concatenate()([conv6,deco6])
    deco5 = tf.keras.layers.Activation(tf.keras.activation.relu())(deco5)
    deco5 = tf.keras.layers.Conv2DTranspose(filters=512,kernel_size=(4,4),stride=(2,2))(deco5) #8
    deco5 = tf.keras.layers.BatchNormalization()(deco5)

    deco4 = tf.keras.layers.Concatenate()([conv5,deco5])
    deco4 = tf.keras.layers.Activation(tf.keras.activation.relu())(deco4)
    deco4 = tf.keras.layers.Conv2DTranspose(filters=512,kernel_size=(4,4),stride=(2,2))(deco4) #16
    deco4 = tf.keras.layers.BatchNormalization()(deco4)

    deco3 = tf.keras.layers.Concatenate()([conv4,deco4])
    deco3 = tf.keras.layers.Activation(tf.keras.activation.relu())(deco3)
    deco3 = tf.keras.layers.Conv2DTranspose(filters=256,kernel_size=(4,4),stride=(2,2))(deco3) #32
    deco3 = tf.keras.layers.BatchNormalization()(deco3)

    deco2 = tf.keras.layers.Concatenate()([conv3,deco3])
    deco2 = tf.keras.layers.Activation(tf.keras.activation.relu())(deco2)
    deco2 = tf.keras.layers.Conv2DTranspose(filters=128,kernel_size=(4,4),stride=(2,2))(deco2) #64
    deco2 = tf.keras.layers.BatchNormalization()(deco2)

    deco1 = tf.keras.layers.Concatenate()([conv2,deco2])
    deco1 = tf.keras.layers.Activation(tf.keras.activation.relu())(deco1)
    deco1 = tf.keras.layers.Conv2DTranspose(filters=64,kernel_size=(4,4),stride=(2,2))(deco1) #128
    deco1 = tf.keras.layers.BatchNormalization()(deco1)

    res = tf.keras.layers.Concatenate()([conv1,deco1])
    res = tf.keras.layers.Activation(tf.keras.activation.relu())(res)
    res = tf.keras.layers.Conv2DTranspose(filters=output_channel,kernel_size=(4,4),stride=(2,2))(res) #256

    inpt_ave = tf.math.reduce_mean(inpt,axis=-1)
    outpt = tf.keras.layers.Add()([res,inpt_ave])
    
    model = tf.keras.Model(inputs=inpt,outputs=outpt)
    return model


def UNet_deconv(image_channels=12,output_channel=2):
    """
    Original U-Net architecture.
    [1] Liu, F., Kijowski, R., Feng, L. & El Fakhri, G. 
        High-performance rapid MR parameter mapping using model-based deep adversarial learning. 
        Magn. Reson. Imaging 74, 152–160 (2020).
    """
    inpt  = tf.keras.layers.Input(shape=(256,256,image_channels)) # 12
    
    conv1 = tf.keras.layers.Conv2D(filters=64,kernel_size=(4,4),stride=(2,2))(inpt) #128
    conv1 = tf.keras.layers.BatchNormalization()(conv1)

    conv2 = tf.keras.layers.Activation(tf.keras.activation.relu(alpha=0.2))(conv1)
    conv2 = tf.keras.layers.Conv2D(filters=128,kernel_size=(4,4),stride=(2,2))(conv1) #64
    conv2 = tf.keras.layers.BatchNormalization()(conv2)

    conv3 = tf.keras.layers.Activation(tf.keras.activation.relu(alpha=0.2))(conv2)
    conv3 = tf.keras.layers.Conv2D(filters=256,kernel_size=(4,4),stride=(2,2))(conv3) #32
    conv3 = tf.keras.layers.BatchNormalization()(conv3)

    conv4 = tf.keras.layers.Activation(tf.keras.activation.relu(alpha=0.2))(conv3)
    conv4 = tf.keras.layers.Conv2D(filters=512,kernel_size=(4,4),stride=(2,2))(conv4) #16
    conv4 = tf.keras.layers.BatchNormalization()(conv4)

    conv5 = tf.keras.layers.Activation(tf.keras.activation.relu(alpha=0.2))(conv4)
    conv5 = tf.keras.layers.Conv2D(filters=512,kernel_size=(4,4),stride=(2,2))(conv5) #8
    conv5 = tf.keras.layers.BatchNormalization()(conv5)

    conv6 = tf.keras.layers.Activation(tf.keras.activation.relu(alpha=0.2))(conv5)
    conv6 = tf.keras.layers.Conv2D(filters=512,kernel_size=(4,4),stride=(2,2))(conv6) #4
    conv6 = tf.keras.layers.BatchNormalization()(conv6)

    conv7 = tf.keras.layers.Activation(tf.keras.activation.relu(alpha=0.2))(conv6)
    conv7 = tf.keras.layers.Conv2D(filters=512,kernel_size=(4,4),stride=(2,2))(conv7) #2
    conv7 = tf.keras.layers.BatchNormalization()(conv7)

    base = tf.keras.layers.Activation(tf.keras.activation.relu(alpha=0.2))(conv7)
    base = tf.keras.layers.Conv2D(filters=512,kernel_size=(4,4),stride=(2,2))(base) #1
    base = tf.keras.layers.BatchNormalization()(base)
    
    deco7 = tf.keras.layers.Activation(tf.keras.activation.relu())(base)
    deco7 = tf.keras.layers.Conv2DTranspose(filters=512,kernel_size=(4,4),stride=(2,2))(deco7) #2
    deco7 = tf.keras.layers.BatchNormalization()(deco7)

    deco6 = tf.keras.layers.Concatenate()([conv7,deco7])
    deco6 = tf.keras.layers.Activation(tf.keras.activation.relu())(deco6)
    deco6 = tf.keras.layers.Conv2DTranspose(filters=512,kernel_size=(4,4),stride=(2,2))(deco6) #4
    deco6 = tf.keras.layers.BatchNormalization()(deco6)

    deco5 = tf.keras.layers.Concatenate()([conv6,deco6])
    deco5 = tf.keras.layers.Activation(tf.keras.activation.relu())(deco5)
    deco5 = tf.keras.layers.Conv2DTranspose(filters=512,kernel_size=(4,4),stride=(2,2))(deco5) #8
    deco5 = tf.keras.layers.BatchNormalization()(deco5)

    deco4 = tf.keras.layers.Concatenate()([conv5,deco5])
    deco4 = tf.keras.layers.Activation(tf.keras.activation.relu())(deco4)
    deco4 = tf.keras.layers.Conv2DTranspose(filters=512,kernel_size=(4,4),stride=(2,2))(deco4) #16
    deco4 = tf.keras.layers.BatchNormalization()(deco4)

    deco3 = tf.keras.layers.Concatenate()([conv4,deco4])
    deco3 = tf.keras.layers.Activation(tf.keras.activation.relu())(deco3)
    deco3 = tf.keras.layers.Conv2DTranspose(filters=256,kernel_size=(4,4),stride=(2,2))(deco3) #32
    deco3 = tf.keras.layers.BatchNormalization()(deco3)

    deco2 = tf.keras.layers.Concatenate()([conv3,deco3])
    deco2 = tf.keras.layers.Activation(tf.keras.activation.relu())(deco2)
    deco2 = tf.keras.layers.Conv2DTranspose(filters=128,kernel_size=(4,4),stride=(2,2))(deco2) #64
    deco2 = tf.keras.layers.BatchNormalization()(deco2)

    deco1 = tf.keras.layers.Concatenate()([conv2,deco2])
    deco1 = tf.keras.layers.Activation(tf.keras.activation.relu())(deco1)
    deco1 = tf.keras.layers.Conv2DTranspose(filters=64,kernel_size=(4,4),stride=(2,2))(deco1) #128
    deco1 = tf.keras.layers.BatchNormalization()(deco1)

    outpt = tf.keras.layers.Concatenate()([conv1,deco1])
    outpt = tf.keras.layers.Activation(tf.keras.activation.relu())(outpt)
    outpt = tf.keras.layers.Conv2DTranspose(filters=output_channel,kernel_size=(4,4),stride=(2,2))(outpt) #256
    
    model = tf.keras.Model(inputs=inpt,outputs=outpt)
    return model