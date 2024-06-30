import tensorflow as tf

class CNN(tf.keras.layers.Layer):
    def __init__(self,Nconv=7,Np=2,filters=64,f=3,name=None):
        super(CNN,self).__init__(name=name)
        self.Nconv = Nconv
        self.conv  = {}
        for i in range(1,self.Nconv): 
            self.conv[i] = tf.keras.layers.Conv2D(filters=filters,kernel_size=(f,f),padding='same',kernel_initializer='he_normal',name='conv'+str(i))
        self.conv[self.Nconv] = tf.keras.layers.Conv2D(filters=Np,kernel_size=(f,f),padding='same',kernel_initializer='he_normal',name='conv'+str(self.Nconv))
        
    def call(self,inputs):
        x = inputs
        x = self.conv[1](x)
        x = tf.keras.layers.ReLU()(x)

        for i in range(2,self.Nconv):
            x = self.conv[i](x)
            x = tf.keras.layers.ReLU()(x)

        x = self.conv[self.Nconv](x)
        x = tf.math.subtract(inputs,x)
        return x

class CNN_Dilate(tf.keras.layers.Layer):
    '''
    Receptive field of size 33*33.
    '''
    def __init__(self,Nconv=7,Np=2,filters=64,f=3,norm=True,name=None):
        super(CNN_Dilate,self).__init__(name=name)
        self.Nconv = Nconv
        self.conv  = {}
        self.norm  = norm
        dr = [1,2,3,4,3,2,1]
        for i in range(1,self.Nconv): 
            self.conv[i] = tf.keras.layers.Conv2D(filters=filters,kernel_size=(f,f),padding='same',dilation_rate=(dr[i-1],dr[i-1]),kernel_initializer='he_normal',name='conv'+str(i))
        self.conv[self.Nconv] = tf.keras.layers.Conv2D(filters=Np,kernel_size=(f,f),padding='same',dilation_rate=(dr[-1],dr[-1]),kernel_initializer='he_normal',name='conv'+str(self.Nconv))
        
        if self.norm == True:
            self.bn = {}
            for i in range(2,self.Nconv): self.bn[i] = tf.keras.layers.BatchNormalization(axis=-1,name='bn'+str(i))

    def call(self,inputs):
        x = inputs
        x = self.conv[1](x)
        x = tf.keras.layers.ReLU()(x)

        for i in range(2,self.Nconv):
            x = self.conv[i](x)
            if self.norm == True: x = self.bn[i](x)
            x = tf.keras.layers.ReLU()(x)

        x = self.conv[self.Nconv](x)
        x = tf.math.subtract(inputs,x)
        return x

class ResBlock(tf.keras.layers.Layer):
    # 258,368 parameters.
    def __init__(self,filters=64,f=3,name=None):
        super(ResBlock,self).__init__(name=name)
        self.conv = {}
        for i in range(5): 
            self.conv[i] = tf.keras.layers.Conv2D(filters=filters,kernel_size=(f,f),padding='same',kernel_initializer='he_normal',name='conv_'+str(i))

    def call(self,inputs):
        x1 = inputs

        x2 = self.conv[0](x1)
        x2 = tf.keras.layers.Activation('relu')(x2)
        x2 = self.conv[1](x2)
        x2 = tf.keras.layers.Add()([x1,x2])
        x2 = tf.keras.layers.Activation('relu')(x2)

        x3 = self.conv[2](x2)
        x3 = tf.keras.layers.Activation('relu')(x3)
        x3 = self.conv[3](x3)
        x3 = tf.keras.layers.Add()([x2,x3])
        x3 = tf.keras.layers.Activation('relu')(x3)

        x4 = tf.keras.layers.Concatenate()([x3,x2,x1])
        x4 = self.conv[4](x4)
        x4 = tf.keras.layers.Activation('relu')(x4)
        return x4

class MOLED(tf.keras.layers.Layer):
    def __init__(self,Nblock=4,Np=2,filters=64,f=3,name=None):
        super(MOLED,self).__init__(name=name)
        self.Nblock    = Nblock
        self.conv_in   = tf.keras.layers.Conv2D(filters=filters,kernel_size=(f,f),padding='same',kernel_initializer='he_normal',name='conv_in')
        self.conv_out  = tf.keras.layers.Conv2D(filters=Np,kernel_size=(f,f),padding='same',kernel_initializer='he_normal',name='conv_out')
        self.res_block = {}
        for i in range(self.Nblock): 
            self.res_block[i] = ResBlock(filters=filters,f=f,name='resblock_'+str(i))

    def call(self,inputs):
        x = inputs
        x = self.conv_in(x)
        x = tf.keras.layers.Activation('relu')(x)
        for i in range(self.Nblock): 
            x = self.res_block[i](x)
        x = self.conv_out(x)
        x = tf.keras.layers.Activation('relu')(x)
        return x

class DnCNN(tf.keras.layers.Layer):
    '''
    Receptive field of size 41*41.
    '''
    def __init__(self,Nconv=20,Np=2,filters=64,f=3,norm=True,name=None):
        super(DnCNN,self).__init__(name=name)
        self.Nconv = Nconv
        self.conv  = {} # convolution
        self.norm  = norm
        for i in range(1,self.Nconv): 
            self.conv[i] = tf.keras.layers.Conv2D(filters=filters,kernel_size=(f,f),padding='same',kernel_initializer='he_normal',name='conv'+str(i))
        self.conv[self.Nconv] = tf.keras.layers.Conv2D(filters=Np,kernel_size=(f,f),padding='same',kernel_initializer='he_normal',name='conv'+str(self.Nconv))

        if self.norm == True:
            self.bn = {}
            # for i in range(2,self.Nconv): self.bn[i] = tf.keras.layers.BatchNormalization(axis=-1,name='bn'+str(i))
            for i in range(2,self.Nconv): self.bn[i] = tf.keras.layers.LayerNormalization(axis=-1,name='bn'+str(i))

    def call(self,inputs):
        x = inputs
        x = self.conv[1](x)
        x = tf.keras.layers.ReLU()(x)

        for i in range(2,self.Nconv):
            x = self.conv[i](x)
            if self.norm == True: x = self.bn[i](x)
            x = tf.keras.layers.ReLU()(x)

        x = self.conv[self.Nconv](x)
        x = tf.math.subtract(inputs,x)
        return x