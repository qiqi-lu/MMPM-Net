# Original UNet
from os import name
from config import config_gpu
from custom_layers import upsample_deconv, Upsample_upconv
import tensorflow as tf

def UNet(input_channels=12,output_channels=2,upsample_mode='upconv'):
    """ Ref: 1. Ronneberger O, Fischer P, Brox T. U-net: Convolutional networks for biomedical image segmentation (2015)
    """

    if upsample_mode == 'deconv':
        upsample = upsample_deconv
    elif upsample_mode == 'upconv':
        upsample = Upsample_upconv
    else:
        print(':: Error: please set the correct upsample mode.')

    inpt = tf.keras.layers.Input(shape=(None,None,input_channels)) # network input
    conv1 = tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same')(inpt)
    conv1 = tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same')(conv1)

    pool1 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(conv1)
    conv2 = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(pool1)
    conv2 = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(conv2)

    pool2 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(conv2)
    conv3 = tf.keras.layers.Conv2D(256,(3,3),activation='relu',padding='same')(pool2)
    conv3 = tf.keras.layers.Conv2D(256,(3,3),activation='relu',padding='same')(conv3)
    
    pool3 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(conv3)
    conv4 = tf.keras.layers.Conv2D(512,(3,3),activation='relu',padding='same')(pool3)
    conv4 = tf.keras.layers.Conv2D(512,(3,3),activation='relu',padding='same')(conv4)

    pool4 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(conv4)
    convbase = tf.keras.layers.Conv2D(1024,(3,3),activation='relu',padding='same')(pool4)
    convbase = tf.keras.layers.Conv2D(1024,(3,3),activation='relu',padding='same')(convbase)
    
    # conc5 = tf.keras.layers.Concatenate()([tf.keras.layers.Conv2D(512,(3,3),padding='same')(tf.keras.layers.UpSampling2D(size=(2,2))(convbase)),conv4])
    conc5 = tf.keras.layers.Concatenate()([upsample(512,(2,2),strides=(2,2),padding='same')(convbase),conv4])
    conv5 = tf.keras.layers.Conv2D(512,(3,3),activation='relu',padding='same')(conc5) 
    conv5 = tf.keras.layers.Conv2D(512,(3,3),activation='relu',padding='same')(conv5) 
    
    # conc6 = tf.keras.layers.Concatenate()([tf.keras.layers.Conv2D(256,(3,3),padding='same')(tf.keras.layers.UpSampling2D(size=(2,2))(conv5)),conv3])
    conc6 = tf.keras.layers.Concatenate()([upsample(256,(2,2),strides=(2,2),padding='same')(conv5),conv3])
    conv6 = tf.keras.layers.Conv2D(256,(3,3),activation='relu',padding='same')(conc6)
    conv6 = tf.keras.layers.Conv2D(256,(3,3),activation='relu',padding='same')(conv6)
    
    # conc7 = tf.keras.layers.Concatenate()([tf.keras.layers.Conv2D(128,(3,3),padding='same')(tf.keras.layers.UpSampling2D(size=(2,2))(conv6)),conv2])
    conc7 = tf.keras.layers.Concatenate()([upsample(128,(2,2),strides=(2,2),padding='same')(conv6),conv2])
    conv7 = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(conc7)
    conv7 = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(conv7)
    
    # conc8 = tf.keras.layers.Concatenate()([tf.keras.layers.Conv2D(64, (3,3),padding='same')(tf.keras.layers.UpSampling2D(size=(2,2))(conv7)),conv1])
    conc8 = tf.keras.layers.Concatenate()([upsample(64,(2,2),strides=(2,2),padding='same')(conv7),conv1])
    conv8 = tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same')(conc8)
    conv8 = tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same')(conv8)
    outpt = tf.keras.layers.Conv2D(output_channels,(1,1),activation='relu',padding='same')(conv8)

    model = tf.keras.Model(inputs=inpt,outputs=outpt)
    return model


def SeparableCNN(depth=8,depth_multi=10,filters=40, image_channels=12,dilation=1):
    """ Ref: 1. Imamura R, Itasaka T, Okuda M. Zero-shot hyperspectral image denoising with separable image prior. 
        Proc. - 2019 Int. Conf. Comput. Vis. Work. ICCVW 2019 2019:1416–1420 doi: 10.1109/ICCVW.2019.00178.
    """
    inpt = tf.keras.layers.Input(shape=(None,None,image_channels))
    # 1st layer, Separable Conv+relu
    x = tf.keras.layers.SeparableConv2D(filters=filters,kernel_size=(3,3),strides=(1,1),
                                        kernel_initializer='Orthogonal',padding='same',
                                        dilation_rate=(dilation,dilation),depth_multiplier=depth_multi)(inpt)
    x = tf.keras.layers.Activation('relu')(x)
    # depth-2 layers, Separable Conv+BN+relu
    for i in range(depth-2):
        x = tf.keras.layers.SeparableConv2D(filters=filters,kernel_size=(3,3),strides=(1,1),
                                            kernel_initializer='Orthogonal',padding='same',
                                            dilation_rate=(dilation,dilation),depth_multiplier=depth_multi)(x)
        x = tf.keras.layers.BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001)(x)
        x = tf.keras.layers.Activation('relu')(x)  
    # last layer, Conv
    x = tf.keras.layers.SeparableConv2D(filters=12,kernel_size=(3,3),strides=(1,1),
                                        kernel_initializer='Orthogonal',padding='same',
                                        dilation_rate=(dilation,dilation),depth_multiplier=depth_multi)(x)
    # ResNet architecture
    outpt = tf.keras.layers.Subtract()([inpt, x])   # input - noise
    model = tf.keras.Model(inputs=inpt, outputs=outpt)
    return model


def UNet_half(image_channels=12):
    """ UNet with half parameters.
    """
    inpt = tf.keras.layers.Input(shape=(None,None,image_channels))
    conv1 = tf.keras.layers.Conv2D(32,(3,3),activation='relu',padding='same')(inpt) # 64*128
    conv1 = tf.keras.layers.Conv2D(32,(3,3),activation='relu',padding='same')(conv1)  # 64*128
    
    pool1 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(conv1) # 32*64
    conv2 = tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same')(pool1)
    conv2 = tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same')(conv2)
    
    pool2 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(conv2) # 16*32
    conv3 = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(pool2)
    conv3 = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(conv3)
    
    pool3 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(conv3) # 8*16
    conv4 = tf.keras.layers.Conv2D(256,(3,3),activation='relu',padding='same')(pool3)
    conv4 = tf.keras.layers.Conv2D(256,(3,3),activation='relu',padding='same')(conv4)
    
    pool4 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(conv4) # 4*8
    convbase = tf.keras.layers.Conv2D(512,(3,3),activation='relu',padding='same')(pool4)    # 4*8
    convbase = tf.keras.layers.Conv2D(512,(3,3),activation='relu',padding='same')(convbase) # 4*8
    
    conc5 = tf.keras.layers.Concatenate()([tf.keras.layers.Conv2D(256,(3,3),padding='same')(tf.keras.layers.UpSampling2D(size=(2,2))(convbase)),conv4]) # 8*16*1024
    conv5 = tf.keras.layers.Conv2D(256,(3,3),activation='relu',padding='same')(conc5) # 8*16*512
    conv5 = tf.keras.layers.Conv2D(256,(3,3),activation='relu',padding='same')(conv5) # 8*16*512
    
    conc6 = tf.keras.layers.Concatenate()([tf.keras.layers.Conv2D(128,(3,3),padding='same')(tf.keras.layers.UpSampling2D(size=(2,2))(conv5)),conv3]) # 16*32*512
    conv6 = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(conc6) # 16*32*256
    conv6 = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(conv6) # 16*32*256
    
    conc7 = tf.keras.layers.Concatenate()([tf.keras.layers.Conv2D(64,(3,3),padding='same')(tf.keras.layers.UpSampling2D(size=(2,2))(conv6)),conv2]) # 32*64*256
    conv7 = tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same')(conc7) # 32*64*128
    conv7 = tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same')(conv7) # 32*64*128
    
    conc8 = tf.keras.layers.Concatenate()([tf.keras.layers.Conv2D(32, (3,3),padding='same')(tf.keras.layers.UpSampling2D(size=(2,2))(conv7)),conv1]) # 64*128*128
    conv8 = tf.keras.layers.Conv2D(32,(3,3),activation='relu',padding='same')(conc8) # 64*128*64
    conv8 = tf.keras.layers.Conv2D(32,(3,3),activation='relu',padding='same')(conv8) # 64*128*64
    outpt = tf.keras.layers.Conv2D(2,(1,1),activation='relu')(conv8) # 64*128*3
    
    model = tf.keras.Model(inputs=inpt,outputs=outpt)
    return model

if __name__ == '__main__':
    import config
    config.config_gpu(1)
    model = UNet(upsample_mode='upconv')
    # model = UNet(upsample_mode='deconv')
    # model = SeparableCNN()
    model = UNet_half()
    y = model(tf.ones(shape=(1,128,128,12)))
    model.summary()

