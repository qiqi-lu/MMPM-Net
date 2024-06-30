import tensorflow as tf

class Jr(tf.keras.layers.Layer):
    def __init__(self,name=None):
        super(Jr,self).__init__(name=name)

    def call(self,x,b,tes):
        m0,p2 = tf.split(x,2,axis=-1)
        tes = tes[:,tf.newaxis,tf.newaxis,:]
        s = tf.math.multiply(m0,tf.math.exp(tf.math.negative(tf.math.multiply(tes,p2))))
        r = tf.math.subtract(s,b) # residual

        dSdm0 = tf.math.exp(tf.math.negative(tf.math.multiply(tes,p2)))
        dSdp2 = tf.math.negative(tf.math.multiply(tes,tf.math.multiply(m0,dSdm0)))

        Jr_m0 = tf.math.reduce_mean(tf.math.multiply(dSdm0,r),axis=-1)
        Jr_p2 = tf.math.reduce_mean(tf.math.multiply(dSdp2,r),axis=-1)
        jacob = tf.stack([Jr_m0,Jr_p2],axis=-1)
        return jacob

class Denoiser(tf.keras.layers.Layer):
    def __init__(self,Nconv=5,Np=1,filters=64,f=3,norm=False,name=None):
        super(Denoiser,self).__init__(name=name)
        self.Nconv = Nconv
        self.norm  = norm
        self.conv  = {}
        for i in range(1,self.Nconv):
            self.conv[i]      = tf.keras.layers.Conv2D(filters=filters,kernel_size=(f,f),strides=(1,1),padding='same',kernel_initializer='he_normal',name='conv'+str(i))
        self.conv[self.Nconv] = tf.keras.layers.Conv2D(filters=Np,kernel_size=(f,f),strides=(1,1),padding='same',kernel_initializer='he_normal',name='conv'+str(self.Nconv))

        if self.norm == True:
            self.Normalization = {}
            for i in range(1,self.Nconv):
                self.Normalization[i] = tf.keras.layers.BatchNormalization(axis=-1,name='normalization_'+str(i))
                # self.Normalization[i] = tf.keras.layers.LayerNormalization(axis=-1,name='normalization_'+str(i))

    def call(self,inputs):
        x = inputs
        for i in range(1,self.Nconv):
            x = self.conv[i](x)
            if self.norm == True: 
                x = self.Normalization[i](x)
            x = tf.keras.layers.ReLU()(x)
        x = self.conv[self.Nconv](x)
        outpt = tf.math.subtract(inputs,x)
        return outpt

def range_constaint(x,m0_max=3.0,p2_max=10.0):
    x     = tf.math.maximum(x,0.0)
    m0,p2 = tf.split(x,2,axis=-1)
    m0    = tf.math.minimum(m0,m0_max)
    p2    = tf.math.minimum(p2,p2_max)
    x     = tf.concat([m0,p2],axis=-1)
    return x

class DOPAMINEm(tf.keras.layers.Layer):
    def __init__(self,Ns=10,path=2,norm=False,name=None):
        super(DOPAMINEm,self).__init__(name=name)
        self.Ns = Ns
        self.Jr = Jr(name='Jr')
        self.path = path
        self.mu,self.lm = [],[]
        self.D,self.Dm,self.Dp = {},{},{}

        mu_0, lm_0 = 0.01, 0.01
        self.mu = tf.Variable(initial_value=mu_0*tf.ones(self.Ns),trainable=True,name='mu',constraint=tf.keras.constraints.NonNeg())
        self.lm = tf.Variable(initial_value=lm_0*tf.ones(self.Ns),trainable=True,name='lm',constraint=tf.keras.constraints.NonNeg())
        for i in range(self.Ns):
            if self.path == 1: self.D[i] = Denoiser(Nconv=7,Np=2,filters=64,f=3,norm=norm,name='D_'+str(i))
            if self.path == 2:
                self.Dm[i] = Denoiser(Nconv=5,Np=1,filters=64,f=3,norm=norm,name='Dm_'+str(i))
                self.Dp[i] = Denoiser(Nconv=5,Np=1,filters=64,f=3,norm=norm,name='Dp_'+str(i))

    def call(self,inputs):
        b   = inputs[0]
        tes = inputs[1]
        xm  = []

        # Initialization
        m0 = tf.math.reduce_max(b,axis=-1)
        p2 = tf.math.add(tf.math.multiply(b[...,0],0.0),1.0)
        x  = tf.stack([m0,p2],axis=-1)
        x  = range_constaint(x,m0_max=3.0,p2_max=10.0)
        # xm.append(x) #test

        for i in range(300):
            gx = self.Jr(x,b,tes)
            x = tf.math.subtract(x,tf.math.multiply(2.0,gx))
            x = range_constaint(x,m0_max=3.0,p2_max=10.0)
        # xm.append(x)

        for i in range(self.Ns):
            if self.path == 1: Dx = self.D[i](x)
            if self.path == 2:
                x0,x1 = tf.split(x,2,axis=-1)
                x0,x1 = self.Dm[i](x0), self.Dp[i](x1)
                Dx    = tf.keras.layers.Concatenate(axis=-1)([x0,x1])
            Dx = range_constaint(Dx,m0_max=3.0,p2_max=10.0)
            xm.append(Dx) #train

            gx = self.Jr(x,b,tes)
            d = tf.math.add(gx,tf.math.multiply(self.lm[i],tf.math.subtract(x,Dx)))
            x = tf.math.subtract(x,tf.math.multiply(self.mu[i],d))

            x = range_constaint(x,m0_max=3.0,p2_max=10.0)
            xm.append(x) #train/test
        xm.append(x) #train
        return tf.stack(xm)

class DOPAMINEm_sw(tf.keras.layers.Layer):
    def __init__(self,Ns=10,path=1,norm=False,name=None):
        super(DOPAMINEm_sw,self).__init__(name=name)
        self.Ns = Ns
        self.Jr = Jr(name='Jr')
        self.path = path

        mu_0, lm_0 = 0.01, 0.01
        self.mu = tf.Variable(initial_value=mu_0*tf.ones(self.Ns),trainable=True,name='mu',constraint=tf.keras.constraints.NonNeg())
        # self.lm = tf.Variable(initial_value=lm_0*tf.ones(self.Ns),trainable=True,name='lm',constraint=tf.keras.constraints.NonNeg())
        self.lm = tf.Variable(initial_value=lm_0,trainable=True,name='lm',constraint=tf.keras.constraints.NonNeg())
        if self.path == 1: self.D = Denoiser(Nconv=7,Np=2,filters=64,f=3,norm=norm,name='D')
        if self.path == 2: 
            self.Dm = Denoiser(Nconv=5,Np=1,filters=64,f=3,norm=norm,name='Dm')
            self.Dp = Denoiser(Nconv=5,Np=1,filters=64,f=3,norm=norm,name='Dp')
    
    def call(self,inputs):
        b   = inputs[0]
        tes = inputs[1]
        xm  = []

        # Initialization
        m0 = tf.math.reduce_max(b,axis=-1)
        p2 = tf.math.add(tf.math.multiply(b[...,0],0.0),1.0)
        x  = tf.stack([m0,p2],axis=-1)
        x  = range_constaint(x,m0_max=3.0,p2_max=10.0)
        # xm.append(x) #test

        for i in range(300):
            gx = self.Jr(x,b,tes)
            x = tf.math.subtract(x,tf.math.multiply(2.0,gx))
            x = range_constaint(x,m0_max=3.0,p2_max=10.0)
        # xm.append(x)

        for i in range(self.Ns):
            if self.path == 1: Dx = self.D(x)
            if self.path == 2:
                x0,x1 = tf.split(x,2,axis=-1)
                x0,x1 = self.Dm(x0), self.Dp(x1)
                Dx    = tf.keras.layers.Concatenate(axis=-1)([x0,x1])
            Dx = range_constaint(Dx,m0_max=3.0,p2_max=10.0)
            xm.append(Dx) #train

            gx = self.Jr(x,b,tes)
            # d = tf.math.add(gx,tf.math.multiply(self.lm[i],tf.math.subtract(x,Dx)))
            d = tf.math.add(gx,tf.math.multiply(self.lm,tf.math.subtract(x,Dx)))
            x = tf.math.subtract(x,tf.math.multiply(self.mu[i],d))

            x = range_constaint(x,m0_max=3.0,p2_max=10.0)
            xm.append(x) #train/test
        xm.append(x) #train
        return tf.stack(xm)