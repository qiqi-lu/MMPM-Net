# custom MoDL model
import tensorflow as tf
import numpy as np
import helper

class Denoiser(tf.keras.layers.Layer):
    """
    Residual learning based denoiser.
    """
    def __init__(self,nLayers=5,out=2,BN=True,trainable=True,**kwargs):
        super(Denoiser,self).__init__(**kwargs)
        self.nLay=nLayers
        self.filter={key: 64 for key in np.arange(1,self.nLay)}
        self.filter[self.nLay]= out 
        self.N={}  # estimated noise
        self.conv={}
        self.bn={} # Batch Normalization
        self.BN=BN

        for i in range(1,self.nLay+1):
            self.conv[i]=tf.keras.layers.Conv2D(filters=self.filter[i],kernel_size=(3,3),strides=(1,1),padding='same',kernel_initializer='he_normal',trainable=trainable)
            # self.conv[i]=tf.keras.layers.Conv2D(filters=self.filter[i],kernel_size=(3,3),strides=(1,1),padding='same',kernel_initializer='he_uniform')
            # self.conv[i]=tf.keras.layers.Conv2D(filters=self.filter[i],kernel_size=(3,3),strides=(1,1),padding='same',kernel_initializer='glorot_uniform')
            if self.BN:
                self.bn[i]=tf.keras.layers.BatchNormalization(trainable=trainable)

    def call(self,inputs):
        self.N['c'+str(0)]=inputs
        lastLayer=False
        for i in range(1,self.nLay+1):
            if i==self.nLay:
                lastLayer=True
            with tf.name_scope('Layer'+str(i)):
                x = self.conv[i](self.N['c'+str(i-1)])
                if self.BN: x = self.bn[i](inputs=x) # batch normalization

                if lastLayer:
                    self.N['c'+str(i)] = x
                    lastLayer=False
                else:
                    self.N['c'+str(i)] = tf.keras.layers.Activation('relu')(x)  

        with tf.name_scope('Residual'):
            outpt = tf.keras.layers.add([inputs,self.N['c'+str(self.nLay)]])
        return outpt

    def get_config(self):
        config = super(Denoiser,self).get_config()
        config.update({
            "nLay":self.nLay,
            "filter":self.filter,
            "N":self.N,
            "conv":self.conv,
            "bn":self.bn,
            "BN":self.BN
        })
        return config
    @classmethod
    def from_config(cls,config):
        return cls(**config)

class Mapping(tf.keras.layers.Layer):
    """
    CNN-based mapping network. (DOPAMINE)
    """
    def __init__(self, nLayers=5,out=2,**kwargs):
        super(Mapping,self).__init__(**kwargs)
        self.nLay=nLayers
        self.filter={key: 64 for key in np.arange(1,self.nLay)}
        self.filter[self.nLay]= out 
        self.conv={}
        self.bn={}
        for i in range(1,self.nLay+1):
            self.conv[i] = tf.keras.layers.Conv2D(filters=self.filter[i],kernel_size=(1,1),strides=(1,1),padding='same',kernel_initializer='he_normal')
            self.bn[i]   = tf.keras.layers.BatchNormalization()
    def call(self,inputs):
        x=inputs
        lastLayer=False
        for i in range(1,self.nLay+1):
            if i ==self.nLay:
                lastLayer=True
            x=self.conv[i](x)
            if lastLayer:
                x=x
            else:
                x=self.bn[i](x)
                x=tf.keras.layers.Activation('relu')(x)
        return x
    def get_config(self):
        config = super(Mapping,self).get_config()
        config.update({
            "nLay":self.nLay,
            "filter":self.filter,
            "conv":self.conv,
            "bn":self.bn,
        })
    @classmethod
    def from_config(cls,config):
        return cls(**config)

class A(tf.keras.layers.Layer):
    def __init__(self,TEs=None,**kwargs):
        super(A,self).__init__(**kwargs)
        self.TEs=TEs
        self.nTE=self.TEs.shape[-1]
        
    def call(self,inputs):
        in_shape=tf.shape(inputs)
        s0_vec=tf.reshape(inputs[...,0],[-1,1]) 
        r2_vec=tf.reshape(inputs[...,1],[-1,1])
        s=s0_vec*tf.exp((-1)*self.TEs/1000.0*r2_vec)
        outputs=tf.reshape(s,[in_shape[-4],in_shape[-3],in_shape[-2],self.nTE]) # may need a batch dim
        # outputs=tf.reshape(s,[in_shape[-3],in_shape[-2],self.nTE]) # may need a batch dim
        return outputs
    def get_config(self):
        config = super(A,self).get_config()
        config.update({
            'TEs':self.TEs,
            'nTE':self.nTE
        })
        return config
    @classmethod
    def from_config(cls,config):
        return cls(**config)


class JAT(tf.keras.layers.Layer):
    def __init__(self,TEs=None,**kwargs):
        super(JAT,self).__init__(**kwargs)
        self.TEs=TEs
        self.nTE=self.TEs.shape[-1]

    def call(self,inputs):
        dSds0=[]
        dSdr2=[]
        x=inputs[0]
        b=inputs[1]
        r=A(self.TEs)(x)-b
        for i in range(0,self.nTE):
            dSds0.append(tf.exp((-1)*self.TEs[i]/1000.0*x[...,1])*r[...,i])
            dSdr2.append((-1)*self.TEs[i]/1000.0*x[...,0]*tf.exp((-1)*self.TEs[i]/1000.0*x[...,1])*r[...,i])
        outputs = tf.stack([tf.math.add_n(dSds0),tf.math.add_n(dSdr2)],axis=-1)
        return outputs
    def get_config(self):
        config=super(JAT,self).get_config()
        config.update({
            'TEs':self.TEs,
            'nTE':self.nTE
        })
        return config
    @classmethod
    def from_config(cls,config):
        return cls(**config)

class superPara(tf.keras.layers.Layer):
    def __init__(self,initial_value=0.05,trainable=True,**kwargs):
        super(superPara,self).__init__(**kwargs)
        # self.w=tf.Variable(initial_value=initial_value,trainable=trainable,name=name,constraint=lambda x: tf.math.minimum(tf.math.maximum(x,0),0.25))
        self.w=tf.Variable(initial_value=initial_value,trainable=trainable,constraint=lambda x: tf.math.maximum(x,0))
        # self.w=tf.Variable(initial_value=initial_value,trainable=trainable,name=name)
        # self.w=tf.Variable(initial_value=initial_value,trainable=trainable,name=name,constraint=lambda x: 0.5/(1+tf.exp(-x)))

    def call(self,inputs):
        outputs=self.w*inputs
        return outputs
    def get_config(self):
        config=super(superPara,self).get_config()
        config.update({
            'w':self.w
        })
        return config


def MoDLFixMuMask(nLayers=5,K=1,training=True,tes=None):
    """
    Model based model (fixed miu through iterations).
    """
    # INPUT
    x0 = tf.keras.Input(shape=(None,None,2),name='x0')
    b  = tf.keras.Input(shape=(None,None,12),name='b')
    M  = tf.keras.Input(shape=(None,None,2),name='M')
    

    # DEFINE THE SHARED LAYERS
    Dn = Denoiser(nLayers=nLayers,training=training) # nLayers >=3
    Jaco  = JAT(tes) 
    Mu={}
    Lambda = superPara(initial_value=4.0,trainable=True,name='Lambda')
    
    for k in range(1,K+1):
        Mu[k]=superPara(initial_value=0.01,trainable=True,name='Mu'+str(k)) # Fixed through iteration
    # OUTPUT
    Dc={}

    # DEFINE THE MODEL 
    Dc['dc0']=x0*M
    for k in range(1,K+1):
        x=Dc['dc'+str(k-1)]
        Dx=Dn(x)
        Dx=Dx*M
        Jr=Jaco([x,b])
        tmp=tf.keras.layers.Subtract()([x,Dx]) # residual aliasing and noise-like artifacts
        tmp=tf.keras.layers.Add()([Jr,Lambda(tmp)])
        tmp=tf.keras.layers.Subtract()([x,2*Mu[k](tmp)])
        Dc['dc'+str(k)]=tmp*M
    outputs=Dc['dc'+str(K)]

    model = tf.keras.Model(inputs=[x0,b,M],outputs=outputs)
    model.summary()
    return model



def IterModel(nLayers=5,K=1,training=True,tes=None):
    """
    Only the model block.
    """
    # INPUT
    x0 = tf.keras.Input(shape=(None,None,2),name='x0')
    b  = tf.keras.Input(shape=(None,None,12),name='b')

    # DEFINE THE SHARED LAYERS
    Jaco  = JAT(tes) 
    Mu={}
    for k in range(1,K+1):
        Mu[k] = superPara(initial_value=0.5,trainable=True,name='Mu'+str(k))

    # OUTPUT
    Dc={}

    # DEFINE THE MODEL 
    Dc['dc0']=x0

    for k in range(1,K+1):
        x=Dc['dc'+str(k-1)]
        Jr=Jaco([x,b])
        l3=tf.keras.layers.Subtract()([x,2*Mu[k](Jr)])
        Dc['dc'+str(k)]=l3

    outputs=Dc['dc'+str(K)]

    model = tf.keras.Model(inputs=[x0,b],outputs=outputs)
    model.summary()
    return model

def upsample_deconv(filters,kernel_size, strides, padding):
    return tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)

class Upsample_upconv(tf.keras.layers.Layer):
    """An upsampling of the feature map followed by a 2*2 convolution.
    """

    def __init__(self,filters=64,kernel_size=(2,2),strides=(2,2),padding='same'):
        super(Upsample_upconv,self).__init__()
        self.up = tf.keras.layers.UpSampling2D(size=strides)
        self.conv = tf.keras.layers.Conv2D(filters,kernel_size,padding=padding)

    def call(self,inputs):
        x = self.up(inputs)
        return self.conv(x)