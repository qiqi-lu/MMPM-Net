# residual network
import tensorflow as tf

def ResBlo(x,input_channel=64,num_filters=64):
    conv1   = tf.keras.layers.Conv2D(num_filters,(3,3),padding='same')(x)   # 64
    bn1     = tf.keras.layers.BatchNormalization()(conv1)
    relu1   = tf.keras.layers.Activation('relu')(bn1)

    conv2   = tf.keras.layers.Conv2D(num_filters,(3,3),padding='same')(relu1)   # 64
    bn2     = tf.keras.layers.BatchNormalization()(conv2)
    if input_channel != num_filters:
        x   = tf.keras.layers.Conv2D(num_filters,(1,1),padding='same')(x)
        x   = tf.keras.layers.BatchNormalization()(x)
    add     = tf.keras.layers.Add()([x,bn2])
    x       = tf.keras.layers.Activation('relu')(add)
    return x

def ResNet(image_channels,output_channel=2):
    """
    Modified ResNet architecture. (RIM)
    """
    inpt    = tf.keras.layers.Input(shape=(None,None,image_channels)) # 12

    conv    = tf.keras.layers.Conv2D(40,(1,1),padding='same')(inpt)   # 64
    bn      = tf.keras.layers.BatchNormalization()(conv)
    x       = tf.keras.layers.Activation('relu')(bn)

    num_features        = [40, 40,40,80,80,160,320,160,80,80,40,6]
    num_residual_block  = len(num_features)-1
    for i in range(num_residual_block):
        x   = ResBlo(x,input_channel=num_features[i],num_filters=num_features[i+1])

    x       = tf.keras.layers.Conv2D(output_channel,(1,1),padding='same')(x)   # 64
    model   = tf.keras.Model(inputs=inpt,outputs=x)
    return model

def ResNet_oled(image_channels=6,output_channel=2,num_block=2):
    """
    OLED ResNet architecture.
    """
    inpt    = tf.keras.layers.Input(shape=(None,None,image_channels))

    conv    = tf.keras.layers.Conv2D(64,(1,1),padding='same')(inpt)
    bn      = tf.keras.layers.BatchNormalization()(conv)
    relu    = tf.keras.layers.Activation('relu')(bn)

    for i in range(num_block):
        conv2   = tf.keras.layers.Conv2D(64,(3,3),padding='same')(relu)
        bn2     = tf.keras.layers.BatchNormalization()(conv2)
        relu2   = tf.keras.layers.Activation('relu')(bn2)

        conv3   = tf.keras.layers.Conv2D(64,(3,3),padding='same')(relu2)
        bn3     = tf.keras.layers.BatchNormalization()(conv3)
        
        add1    = tf.keras.layers.Add()([relu,bn3])

        conv4   = tf.keras.layers.Conv2D(64,(3,3),padding='same')(add1)
        bn4     = tf.keras.layers.BatchNormalization()(conv4)
        relu    = tf.keras.layers.Activation('relu')(bn4)

    outpt   = tf.keras.layers.Conv2D(output_channel,(1,1),padding='same')(relu) 
    model   = tf.keras.Model(inputs=inpt,outputs=outpt)
    return model



def ResBlo_moled(x):
    conv1   = tf.keras.layers.Conv2D(64,(3,3),padding='same')(x)
    relu1   = tf.keras.layers.Activation('relu')(conv1)
    conv2   = tf.keras.layers.Conv2D(64,(3,3),padding='same')(relu1)
    esum2   = tf.keras.layers.Add()([x,conv2])
    relu2   = tf.keras.layers.Activation('relu')(esum2)

    conv3   = tf.keras.layers.Conv2D(64,(3,3),padding='same')(relu2)
    relu3   = tf.keras.layers.Activation('relu')(conv3)
    conv4   = tf.keras.layers.Conv2D(64,(3,3),padding='same')(relu3)
    esum4   = tf.keras.layers.Add()([relu2,conv4])
    relu4   = tf.keras.layers.Activation('relu')(esum4)

    conc5   = tf.keras.layers.Concatenate()([relu4,relu2,x])
    conv5   = tf.keras.layers.Conv2D(64,(3,3),padding='same')(conc5)
    x       = tf.keras.layers.Activation('relu')(conv5)
    return x

def ResNet_moled(image_channels=6,output_channel=2,num_block=4):
    """
    Multiple OLED network. (Zhang2019a)
    """
    inpt    = tf.keras.layers.Input(shape=(None,None,image_channels))

    conv_in = tf.keras.layers.Conv2D(64,(1,1),padding='same')(inpt)
    x       = tf.keras.layers.Activation('relu')(conv_in)

    for i in range(num_block):
        x   = ResBlo_moled(x)

    conv_out    = tf.keras.layers.Conv2D(output_channel,(1,1),padding='same',kernel_initializer='he_normal')(x) 
    conv_out    = tf.keras.layers.Activation('relu')(conv_out)
    model       = tf.keras.Model(inputs=inpt,outputs=conv_out)
    return model



def CNN(image_channels=6,output_channel=2):
    inpt_b  = tf.keras.layers.Input(shape=(None,None,image_channels))

    x = tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',kernel_initializer='he_normal',dilation_rate=(1,1))(inpt_b)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',kernel_initializer='he_normal',dilation_rate=(2,2))(x)
    x = tf.keras.layers.BatchNormalization(axis=-1,trainable=True)(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',kernel_initializer='he_normal',dilation_rate=(3,3))(x)
    x = tf.keras.layers.BatchNormalization(axis=-1,trainable=True)(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',kernel_initializer='he_normal',dilation_rate=(4,4))(x)
    x = tf.keras.layers.BatchNormalization(axis=-1,trainable=True)(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',kernel_initializer='he_normal',dilation_rate=(3,3))(x)
    x = tf.keras.layers.BatchNormalization(axis=-1,trainable=True)(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',kernel_initializer='he_normal',dilation_rate=(2,2))(x)
    x = tf.keras.layers.BatchNormalization(axis=-1,trainable=True)(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters=output_channel,kernel_size=(1,1),strides=(1,1),padding='same',kernel_initializer='he_normal',dilation_rate=(1,1))(x)

    model = tf.keras.Model(inputs=inpt_b,outputs=x)
    return model
