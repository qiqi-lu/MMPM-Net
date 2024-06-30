import tensorflow as tf

class RNNCell(tf.keras.layers.Layer):
    def __init__(self,Np=2,filters=36,name=None):
        super(RNNCell,self).__init__(name=name)
        self.filters = filters
        self.convs = {}
        self.GRUs  = {}
        for i in range(3):
            self.convs[i] = tf.keras.layers.Conv2D(filters=filters,kernel_size=(3,3),strides=(1,1),padding='same',use_bias=True,kernel_initializer='he_normal',name='conv'+str(i))
        self.convs[3] = tf.keras.layers.Conv2D(filters=Np,kernel_size=(1,1),strides=(1,1),padding='same',use_bias=True,kernel_initializer='he_normal',name='conv'+str(3))
        
        for j in range(2): self.GRUs[j] = tf.keras.layers.GRU(units=filters,return_sequences=True, return_state=True,name='GRU_'+str(j))

    def call(self,x,gx,h):
        Ny = tf.shape(x)[1]
        Nx = tf.shape(x)[2]

        h0 = h[0]
        h1 = h[1]

        xgx     = tf.keras.layers.Concatenate(axis=-1)([x,gx])

        feature = self.convs[0](xgx)
        feature = tf.keras.layers.Activation('relu')(feature)

        feature     = tf.keras.layers.Reshape(target_shape=(-1,self.filters))(feature)
        feature, h0 = self.GRUs[0](inputs=feature,initial_state=h0)
        feature     = tf.keras.layers.Reshape(target_shape=(Ny,Nx,self.filters))(feature)

        feature = self.convs[1](feature)
        feature = tf.keras.layers.Activation('relu')(feature)

        feature = self.convs[2](feature)
        feature = tf.keras.layers.Activation('relu')(feature)

        feature     = tf.keras.layers.Reshape(target_shape=(-1,self.filters))(feature)
        feature, h1 = self.GRUs[1](inputs=feature,initial_state=h1)
        feature     = tf.keras.layers.Reshape(target_shape=(Ny,Nx,self.filters))(feature)

        dx = self.convs[3](feature)
        hh = [h0,h1]
        return dx,hh

class DataConsistency(tf.keras.layers.Layer):
    def __init__(self,name=None):
        super(DataConsistency,self).__init__(name=name)
    def call(self,x,b,tes):
        tes = tes[:,tf.newaxis,tf.newaxis,:]
        m0,p2 = tf.split(x,2,axis=-1)
        m0 = tf.math.abs(m0)
        p2 = tf.math.abs(p2)
        s  = tf.math.multiply(m0,tf.math.exp(tf.math.negative(tf.math.multiply(tes,p2))))
        # dc = tf.math.reduce_sum(tf.math.square(s-b))
        dc = tf.math.reduce_mean(tf.math.square(s-b),axis=-1)
        dc = tf.math.reduce_sum(dc)
        return dc

def range_constaint(x,m0_max=3.0,p2_max=10.0):
    x     = tf.math.maximum(x,0.0)
    m0,p2 = tf.split(x,2,axis=-1)
    m0    = tf.math.minimum(m0,m0_max)
    p2    = tf.math.minimum(p2,p2_max)
    x     = tf.concat([m0,p2],axis=-1)
    return x

class RIMm(tf.keras.layers.Layer):
    def __init__(self,Ns=6,Np=2,filters=36,name=None):
        super(RIMm,self).__init__(name=name)
        self.Ns = Ns
        self.RNNCell = {}
        self.DC = DataConsistency(name='DC')
        for i in range(self.Ns):
            self.RNNCell[i] = RNNCell(Np=Np,filters=filters,name='RNNCell'+str(i))
    def call(self,inputs):
        b   = inputs[0]
        tes = inputs[1]

        ##### Initialization #####
        m0 = tf.math.reduce_max(b,axis=-1)
        p2 = tf.math.add(tf.math.multiply(b[...,0],0.0),1.0)
        x  = tf.stack([m0,p2],axis=-1)
        x = range_constaint(x,m0_max=3.0,p2_max=10.0)
        
        h  = [None,None]
        xm = []
        # xm.append(x) # test

        ##### Iterations #####
        for i in range(self.Ns):
            with tf.GradientTape() as g:
                g.watch(x)
                dc = self.DC(x,b,tes)
                gx = g.gradient(dc,x)
            gx = tf.where(tf.math.is_nan(gx),0.0,gx)
            gx = tf.where(tf.math.is_inf(gx),0.0,gx)

            dx,h  = self.RNNCell[i](x,gx,h)
            x = tf.math.abs(tf.math.add(x,dx))

            xm.append(x) # train/test
        return tf.stack(xm)
  