import tensorflow as tf
from . import auxiliary_functions as afunc

class Denoiser(tf.keras.layers.Layer):
    def __init__(self,Nconv=7,Np=1,filters=64,f=3,norm=False,name=None):
        super(Denoiser,self).__init__(name=name)
        self.Nconv = Nconv # number of convolutional layer.
        self.norm  = norm  # batch normalization, optional.
        self.conv  = {}    # convolutional layer
        for i in range(1,self.Nconv):
            self.conv[i]      = tf.keras.layers.Conv2D(filters=filters,kernel_size=(f,f),strides=(1,1),padding='same',kernel_initializer='he_normal',name='conv'+str(i))
        self.conv[self.Nconv] = tf.keras.layers.Conv2D(filters=Np,kernel_size=(f,f),strides=(1,1),padding='same',kernel_initializer='he_normal',name='conv'+str(self.Nconv))

        if norm == True:
            self.Normalization = {}
            for i in range(1,self.Nconv): self.Normalization[i] = tf.keras.layers.BatchNormalization(axis=-1,name='normalization_'+str(i))

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

class DataConsistency(tf.keras.layers.Layer):
    def __init__(self,signed=True,name=None):
        super(DataConsistency,self).__init__(name=name)
        self.signed = signed # signed/unsigned magnitude data
    def call(self,x,b,tau):
        tau0 = tau[0] # [Nq]
        s = afunc.physical_model(x=x,tau=tau0,signed=self.signed) # [Nb,Ny,Nx,Nq]
        # dc = tf.math.reduce_sum(tf.math.square(tf.math.subtract(s,b))) # without mean
        dc = tf.math.reduce_mean(tf.math.square(tf.math.subtract(s,b)),axis=-1) # mean along tau dimention, [Nb,Ny,Nx,Nq]
        dc = tf.math.multiply(0.5,tf.math.reduce_sum(dc))
        return dc

class DOPAMINE(tf.keras.layers.Layer):
    def __init__(self,Ns=10,norm=False,signed=True,share_weight=False,ini_mu=0.01,ini_lm=0.01,sep=False,test_mode=True,name=None):
        super(DOPAMINE,self).__init__(name=name)
        self.DC = DataConsistency(signed=signed,name='DC')  # data consistency term
        self.Ns = Ns                # number of stages
        self.sw = share_weight      # share weigths between different stages
        self.sep = sep              # whether to process each channel seperately
        self.tmode = test_mode

        # hyperparameters, lm represents 2*lambda in equation.
        self.mu = tf.Variable(initial_value=ini_mu*tf.ones(self.Ns),trainable=True,name='mu',constraint=tf.keras.constraints.NonNeg())
        self.lm = tf.Variable(initial_value=ini_lm*tf.ones(self.Ns),trainable=True,name='lm',constraint=tf.keras.constraints.NonNeg())
        
        # denoisers
        if share_weight == False:
            self.D = {} # denoisers
            self.D0, self.D1, self.D2 = {}, {}, {}
            for i in range(self.Ns):
                if self.sep == False: 
                    self.D[i] = Denoiser(Nconv=7,Np=3,filters=64,f=3,norm=norm,name='D_'+str(i))
                if self.sep == True:
                    self.D0[i] = Denoiser(Nconv=5,Np=1,filters=64,f=3,norm=norm,name='D0_'+str(i))
                    self.D1[i] = Denoiser(Nconv=5,Np=1,filters=64,f=3,norm=norm,name='D1_'+str(i))
                    self.D2[i] = Denoiser(Nconv=5,Np=1,filters=64,f=3,norm=norm,name='D2_'+str(i))

        if share_weight == True:
            if self.sep == False:
                self.D = Denoiser(Nconv=7,Np=3,filters=64,f=3,norm=norm,name='D')
            if self.sep == True:
                self.D0 = Denoiser(Nconv=5,Np=1,filters=64,f=3,norm=norm,name='D0')
                self.D1 = Denoiser(Nconv=5,Np=1,filters=64,f=3,norm=norm,name='D1')
                self.D2 = Denoiser(Nconv=5,Np=1,filters=64,f=3,norm=norm,name='D2')

    def call(self,inputs):
        b   = inputs[0]
        tau = inputs[1]
        xm  = [] # collect all stage output

        ######## Initialization ########
        A  = tf.math.reduce_max(tf.math.abs(b),axis=-1)
        B  = tf.math.multiply(2.0,A)
        R1 = tf.math.multiply(tf.ones_like(A),1.0)
        x  = tf.stack([A,B,R1],axis=-1)
        x  = afunc.range_constaint_R1(x,A_max=3.0,B_max=6.0,R1_max=50.0)

        if self.tmode == True: xm.append(x) # test

        ##################
        # better initialization
        for i in range(300):
            # Automatic differentiation
            with tf.GradientTape() as g:
                g.watch(x)
                dc = self.DC(x,b,tau)
            gx = g.gradient(dc,x)
            # Replace nan and inf with zero
            gx = tf.where(tf.math.is_nan(gx),0.0,gx)
            gx = tf.where(tf.math.is_inf(gx),0.0,gx)
            # Gradeint descent with a step size 2.0.
            x = tf.math.subtract(x,tf.math.multiply(2.0,gx))
            x = afunc.range_constaint_R1(x,A_max=3.0,B_max=6.0,R1_max=50.0)

        if self.tmode == True: xm.append(x) # test

        ######## Main Iterations ########
        for i in range(self.Ns):
            # Denoising
            if self.sw == False:
                if self.sep == False: 
                    Dx = self.D[i](x)
                if self.sep == True:
                    x0,x1,x2 = tf.split(x,3,axis=-1)
                    x0,x1,x2 = self.D0[i](x0),self.D1[i](x1),self.D2[i](x2)
                    Dx = tf.keras.layers.Concatenate(axis=-1)([x0,x1,x2])
            
            if self.sw == True:
                if self.sep == False:
                    Dx = self.D(x)
                if self.sep == True:
                    x0,x1,x2 = tf.split(x,3,axis=-1)
                    x0,x1,x2 = self.D0(x0),self.D1(x1),self.D2(x2)
                    Dx = tf.keras.layers.Concatenate(axis=-1)([x0,x1,x2])

            Dx = afunc.range_constaint_R1(x,A_max=3.0,B_max=6.0,R1_max=50.0)

            if self.tmode == False: xm.append(Dx) # train

            # Automatic differentiation
            with tf.GradientTape() as g: 
                g.watch(x)
                dc = self.DC(x,b,tau)
            gx = g.gradient(dc,x)

            # Gradient descent
            d = tf.math.add(gx,tf.math.multiply(self.lm[i],tf.math.subtract(x,Dx)))
            x = tf.math.subtract(x,tf.math.multiply(self.mu[i],d))
            x = afunc.range_constaint_R1(x,A_max=3.0,B_max=6.0,R1_max=50.0)

            xm.append(x) #train/test
            
        if self.tmode == False: xm.append(x) #train
        return tf.stack(xm)
