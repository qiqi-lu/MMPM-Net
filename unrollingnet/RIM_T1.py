import tensorflow as tf
from . import auxiliary_functions as afunc

class RNNCell(tf.keras.layers.Layer):
    def __init__(self,Np=3,filters=36,name=None):
        super(RNNCell,self).__init__(name=name)
        self.filters = filters  # filters in convolutional layers
        self.convs = {}         # convolutional layers
        self.GRUs  = {}         # Gated Recurrent Units, GRU
        for i in range(3):
            self.convs[i] = tf.keras.layers.Conv2D(filters=filters,kernel_size=(3,3),strides=(1,1),padding='same',use_bias=True,kernel_initializer='he_normal',name='conv'+str(i))
        self.convs[3] = tf.keras.layers.Conv2D(filters=Np,kernel_size=(1,1),strides=(1,1),padding='same',use_bias=True,kernel_initializer='he_normal',name='conv'+str(3))
        
        for j in range(2): self.GRUs[j] = tf.keras.layers.GRU(units=filters,return_sequences=True, return_state=True,name='GRU_'+str(j))

    def call(self,x,gx,h):
        Ny,Nx = tf.shape(x)[1],tf.shape(x)[2]
        h0,h1 = h[0],h[1]

        xgx = tf.keras.layers.Concatenate(axis=-1)([x,gx])  # [Nb,Ny,Nx,Np*2]
        # Convolutional layer
        feature = self.convs[0](xgx)                        # [Nb,Ny,Nx,self.filters]
        feature = tf.keras.layers.Activation('relu')(feature)
        # RNN layer
        feature     = tf.keras.layers.Reshape(target_shape=(-1,self.filters))(feature)      # [Nb,Ny*Nx,self.filters]
        feature, h0 = self.GRUs[0](inputs=feature,initial_state=h0)
        feature     = tf.keras.layers.Reshape(target_shape=(Ny,Nx,self.filters))(feature)   # [Nb,Ny,Nx,self.filters]
        # Convolutional layer
        feature = self.convs[1](feature)
        feature = tf.keras.layers.Activation('relu')(feature)
        # Convolutional layer
        feature = self.convs[2](feature)
        feature = tf.keras.layers.Activation('relu')(feature)
        # RNN layer
        feature     = tf.keras.layers.Reshape(target_shape=(-1,self.filters))(feature)
        feature, h1 = self.GRUs[1](inputs=feature,initial_state=h1)
        feature     = tf.keras.layers.Reshape(target_shape=(Ny,Nx,self.filters))(feature)   # [Nb,Ny,Nx,self.filters]
        # Convolutional layer
        dx = self.convs[3](feature) # [Nb,Ny,Nx,Np]
        hh = [h0,h1]
        return dx,hh

class DataConsistency(tf.keras.layers.Layer):
    def __init__(self,signed=True,name=None):
        super(DataConsistency,self).__init__(name=name)
        self.signed = signed # signed/unsigned magnitude data 
    def call(self,x,b,tau):
        tau0 = tau[0] # [Nq]
        s = afunc.physical_model(x=x,tau=tau0,signed=self.signed) # [Nb,Ny,Nx,Nq]

        # dc = tf.math.reduce_sum(tf.math.square(s-b)) # without mean
        dc = tf.math.reduce_mean(tf.math.square(s-b),axis=-1) # mean along the tau dimension, [Nb,Ny,Nx]
        dc = tf.math.reduce_sum(dc) # scalar
        return dc

class RIM(tf.keras.layers.Layer):
    def __init__(self,Ns=6,Np=3,filters=36,signed=True,test_mode=False,share_weight=False,name=None):
        super(RIM,self).__init__(name=name)
        self.Ns  = Ns # number of stages
        self.RNNCell = {}
        self.DC = DataConsistency(signed=signed,name='DC')
        self.tmode = test_mode

        if share_weight == True:
            for i in range(self.Ns):
                self.RNNCell[i] = RNNCell(Np=Np,filters=filters,name='RNNCell'+str(i)) # different RNNCell in each stage
        
        if share_weight == False:
            self.RNNCell = RNNCell(Np=Np,filters=filters,name='RNNCell') # same RNNCell in every stage [Original RIM]

    def call(self,inputs):
        b   = inputs[0]
        tau = inputs[1]
        h   = [None,None]
        xm  = [] # collect outputs of all stages

        ##### Initialization #####
        A = tf.math.reduce_max(tf.math.abs(b),axis=-1)
        B = tf.math.multiply(2.0,A)
        R1= tf.math.multiply(tf.ones_like(A),1.0)
        x = tf.stack([A,B,R1],axis=-1)
        x = afunc.range_constaint_R1(x,A_max=3.0,B_max=6.0,R1_max=50.0)
        
        if self.tmode == True: xm.append(x) # test

        ##### Iterations #####
        for i in range(self.Ns):
            with tf.GradientTape() as g:
                g.watch(x)
                dc = self.DC(x,b,tau)
                gx = g.gradient(dc,x)
            # replace nan and inf with zero.
            gx = tf.where(tf.math.is_nan(gx),0.0,gx)
            gx = tf.where(tf.math.is_inf(gx),0.0,gx)

            dx,h  = self.RNNCell[i](x,gx,h) # different RNNCell in each stage
            # dx,h  = self.RNNCell(x,gx,h)  # same RNNCell in every stage
            x = tf.math.add(x,dx)
            x = tf.math.abs(x)

            xm.append(x) # train/test
        return tf.stack(xm)
  