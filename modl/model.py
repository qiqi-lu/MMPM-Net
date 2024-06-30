import layers as lay
import tensorflow as tf

def DenoiserModel(nLayers=5,K=1,tes=None):
    """
    Only use the denoise block, without DC. (CNN)
    """
    x0 = tf.keras.Input(shape=(None,None,2),name='x0')
    x=x0
    Dn = lay.Denoiser(nLayers=nLayers) # nLayers >=3
    for i in range(K):
        x=Dn(x)
    model=tf.keras.Model(inputs=x0,outputs=x)
    model.summary()
    return model

# def DOPAMINE(nLayers=5,K=10,tes=None):
    """
    Deep model-based magnetic resonance parameter mapping network (DOPAMINE) (Jun2021)
    CNN(2) + GD (gradient descent)
    ## INPUT
    ### x0
    The initial estimated x
    ### b
    The weighted images
    """
    # INPUT
    x0 = tf.keras.Input(shape=(None,None,2),name='x0')
    b  = tf.keras.Input(shape=(None,None,12),name='b')

    # DEFINE THE SHARED LAYERS
    DRS0   = lay.Denoiser(nLayers=nLayers,out=1,BN=True) # parameter map reconstruction network.
    DRR2   = lay.Denoiser(nLayers=nLayers,out=1,BN=True)
    Jaco   = lay.JAT(tes) 
    Lambda = {}
    Mu     = {}
    for k in range(1,K+1):
        Lambda[k] = lay.superPara(initial_value=4.0, trainable=True,name='Lambda'+str(k))
        Mu[k]     = lay.superPara(initial_value=0.01,trainable=True,name='Mu'+str(k))

    # GD STEP
    x=x0
    for k in range(1,K+1):
        S0  = DRS0(tf.expand_dims(x[...,0],axis=-1))
        R2  = DRR2(tf.expand_dims(x[...,1],axis=-1))
        DRx = tf.keras.layers.Concatenate()([S0,R2])
        Jr  = Jaco([x,b])
        delta = tf.keras.layers.Subtract()([x,DRx]) # residual aliasing and noise-like artifacts
        delta = tf.keras.layers.Add()([Jr,Lambda[k](delta)])
        x     = tf.keras.layers.Subtract()([x,2*Mu[k](delta)])

    model = tf.keras.Model(inputs=[x0,b],outputs=x)
    model.summary()
    return model

# def MoDL(nLayers=5,K=1,tes=None):
    """
    CNN with Gradient Descent Data Consistency (CNN + GD).
    """
    # INPUT
    x0 = tf.keras.Input(shape=(None,None,2),name='x0')
    b  = tf.keras.Input(shape=(None,None,12),name='b')

    # DEFINE THE SHARED LAYERS
    Dn     = lay.Denoiser(nLayers=nLayers) # nLayers >=3
    Jaco   = lay.JAT(tes) 
    Lambda = {}
    Mu     = {}
    
    for k in range(1,K+1):
        Lambda[k] = lay.superPara(initial_value=4.0,trainable=True,name='Lambda'+str(k))
        Mu[k]     = lay.superPara(initial_value=0.01,trainable=True,name='Mu'+str(k))

    # OUTPUT
    Dc={}

    # DEFINE THE MODEL 
    Dc['dc0']=x0
    for k in range(1,K+1):
        x=Dc['dc'+str(k-1)]
        Dx=Dn(x)
        Jr=Jaco([x,b])
        tmp=tf.keras.layers.Subtract()([x,Dx]) # residual aliasing and noise-like artifacts
        tmp=tf.keras.layers.Add()([Jr,Lambda[k](tmp)])
        tmp=tf.keras.layers.Subtract()([x,2*Mu[k](tmp)])
        Dc['dc'+str(k)]=tmp
    outputs=Dc['dc'+str(K)]

    model = tf.keras.Model(inputs=[x0,b],outputs=outputs)
    model.summary()
    return model

def MoDLM(nLayers=5,K=1,tes=None):
    """
    CNN with Gradient Descent Data Consistency (CNN + GD).
    Using mask.
    """
    # INPUT
    x0 = tf.keras.Input(shape=(None,None,2),name='x0')
    b  = tf.keras.Input(shape=(None,None,12),name='b')
    M  = tf.keras.Input(shape=(None,None,2),name='M')

    # DEFINE THE SHARED LAYERS
    Dn     = lay.Denoiser(nLayers=nLayers) # nLayers >=3
    Jaco   = lay.JAT(tes) 
    Lambda = {}
    Mu     = {}

    for k in range(1,K+1):
        Lambda[k] = lay.superPara(initial_value=0.01,trainable=True,name='Lambda'+str(k))
        Mu[k]     = lay.superPara(initial_value=0.01,trainable=True,name='Mu'+str(k))

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
        tmp=tf.keras.layers.Add()([Jr,Lambda[k](tmp)])
        tmp=tf.keras.layers.Subtract()([x,2*Mu[k](tmp)])
        Dc['dc'+str(k)]=tmp*M

        # l1=tf.keras.layers.Subtract()([x, 2*Mu[k](Lambda[k](x))])
        # l2=tf.keras.layers.Add()([l1, 2*Lambda[k](Mu[k](Dx))])
        # l3=tf.keras.layers.Subtract()([l2,2*Mu[k](Jr)])
        # Dc['dc'+str(k)]=l3

    # outputs=Dn(Dc['dc'+str(K)])
    outputs=Dc['dc'+str(K)]

    model = tf.keras.Model(inputs=[x0,b,M],outputs=outputs)
    model.summary()
    return model

# def MoDLFixMu(nLayers=5,K=1,tes=None,BN=True):
    """
    Model based model (fixed miu through iterations).
    ### PARAMETERS
    nLayers: Number of layers of the Denoiser.
    K: Number of iteration.
    tes: Echo time.
    BN: Whether to apply batch normalization.
    ### RETURNS
    model: Keras model.
    """
    # INPUT
    x0 = tf.keras.Input(shape=(None,None,2),name='x0')
    b  = tf.keras.Input(shape=(None,None,12),name='b')
    
    # DEFINE THE SHARED LAYERS
    Dn = lay.Denoiser(nLayers=nLayers,out=2,BN=BN) # nLayers >=3
    Jaco  = lay.JAT(tes) 
    Mu={}
    Lambda = lay.superPara(initial_value=4.0,trainable=True,name='Lambda')
    
    for k in range(1,K+1):
        Mu[k]=lay.superPara(initial_value=0.01,trainable=True,name='Mu'+str(k)) # Fixed through iteration
        
    # OUTPUT
    Dc={}

    # DEFINE THE MODEL 
    Dc['dc0']=x0
    for k in range(1,K+1):
        x=Dc['dc'+str(k-1)]
        Dx=Dn(x)
        Jr=Jaco([x,b])
        tmp=tf.keras.layers.Subtract()([x,Dx]) # residual aliasing and noise-like artifacts
        tmp=tf.keras.layers.Add()([Jr,Lambda(tmp)])
        tmp=tf.keras.layers.Subtract()([x,2*Mu[k](tmp)])
        Dc['dc'+str(k)]=tmp
    outputs=Dc['dc'+str(K)]

    tf.summary.scalar("Lambda",Lambda.w)
    for i in range(1,K+1):
        tf.summary.scalar('Mu'+str(i),Mu[k].w)

    model = tf.keras.Model(inputs=[x0,b],outputs=outputs)
    model.summary()
    return model

class MoDLFixMu(tf.keras.Model):
    """
    Model based model (fixed miu through iterations).
    ### PARAMETERS
    - nLayers: Number of layers of the Denoiser.
    - K: Number of iteration.
    - tes: Echo time.
    - BN: Whether to apply batch normalization.
    ### RETURNS
    - model: Keras model.
    """
    def __init__(self,nLayers=5,K=1,tes=None,BN=True):
        super(MoDLFixMu,self).__init__()
        # # INPUT
        # self.x0 = tf.keras.Input(shape=(None,None,2),name='x0')
        # self.b  = tf.keras.Input(shape=(None,None,12),name='b')
        
        # DEFINE THE SHARED LAYERS
        self.K=K
        self.Dn = lay.Denoiser(nLayers=nLayers,out=2,BN=BN) # nLayers >=3
        self.Jaco  = lay.JAT(tes) 
        self.Mu={}
        self.Lambda = lay.superPara(initial_value=4.0,trainable=True,name='Lambda')
        
        for k in range(1,K+1):
            self.Mu[k]=lay.superPara(initial_value=0.01,trainable=True,name='Mu'+str(k)) # Fixed through iteration
            
        # OUTPUT
        self.Dc={}
    
    def call(self,inputs):
        # DEFINE THE MODEL 
        x0 = inputs[0]
        b  = inputs[1]
        self.Dc['dc0']=x0
        for k in range(1,self.K+1):
            x=self.Dc['dc'+str(k-1)]
            Dx=self.Dn(x)
            Jr=self.Jaco([x,b])
            tmp=tf.keras.layers.Subtract()([x,Dx]) # residual aliasing and noise-like artifacts
            tmp=tf.keras.layers.Add()([Jr,self.Lambda(tmp)])
            tmp=tf.keras.layers.Subtract()([x,2*self.Mu[k](tmp)])
            self.Dc['dc'+str(k)]=tmp
        return self.Dc['dc'+str(self.K)]

    # tf.summary.scalar("Lambda",Lambda.w)
    # for i in range(1,K+1):
    #     tf.summary.scalar('Mu'+str(i),Mu[k].w)

    # model = tf.keras.Model(inputs=[x0,b],outputs=outputs)
    # model.summary()
    # return model


class MoDL(tf.keras.Model):
    """
    Model based model (fixed miu through iterations).
    ### PARAMETERS
    - nLayers: Number of layers of the `Denoiser`.
    - K: Number of iteration.
    - tes: Echo time.
    - Lambda_type: Use `Diff`erent lambda or `Same` lambda for different iterations, or use a specific lambda value for all iteration.
    - BN: Whether to apply batch normalization in `CNN`.
    - OS: Optimization Strategy. Gradient Descent `GD` or Conjugate Gradient `CG`.
    - TS: Training Strategy, Pred-trained Denoiser `PD` or End-to-End Training `ET`.
    - NA: Network Architecture. No Sharing `NS` or With Sharing `WS`.
    ### RETURNS
    - parameter maps.
    """
    def __init__(self,nLayers=5,K=1,tes=None,Lambda_type='Diff',BN=True,OS='GD',TS='ET',NA='WS',**kwargs):
        super(MoDL,self).__init__(**kwargs)
        self.K=K
        self.Lambda_type = Lambda_type
        self.OS = OS
        self.TS = TS
        self.NA = NA

        os_dict = ['GD','CG']
        ts_dict = ['ET','PD']
        na_dict = ['WS','NS']
        
        assert OS in os_dict, 'Unsupported Optimization Strategy.'
        assert TS in ts_dict, 'Unsupported Training Strategy.'
        assert NA in na_dict, 'Unsupported Network Architecture.'

        if TS=='PD': trainable=False
        if TS=='ET': trainable=True

        if NA=='WS':self.Dn = lay.Denoiser(nLayers=nLayers,out=2,BN=BN,trainable=trainable,name='Denoiser') # nLayers >=3
        if NA=='NS':
            self.Dn={}
            for k in range(1,K+1):
                self.Dn[k] = lay.Denoiser(nLayers=nLayers,out=2,BN=BN,trainable=trainable,name='Denoiser'+str(k)) # nLayers >=3

        self.Jaco  = lay.JAT(tes,name='Jr')

        # SUPER PARAMETERS
        if Lambda_type == 'Same': self.Lambda = lay.superPara(initial_value=4.0,trainable=True,name='Lambda')
        if Lambda_type == 'Diff':
            self.Lambda = {}
            for k in range(1,K+1):
                self.Lambda[k] = lay.superPara(initial_value=4.0,trainable=True,name='Lambda'+str(k))
        if type(Lambda_type) == float or int: 
            self.Lambda = lay.superPara(initial_value=Lambda_type,trainable=False,name='Lambda')

        self.Mu={}
        for k in range(1,K+1):
            self.Mu[k]=lay.superPara(initial_value=0.01,trainable=True,name='Mu'+str(k))
            
        # OUTPUT
        self.Dc={}
    
    def call(self,inputs):
        x0 = inputs[0]
        b  = inputs[1]
        self.Dc['dc0']=x0

        for k in range(1,self.K+1):
            x=self.Dc['dc'+str(k-1)]

            if self.NA=='WS': Dx=self.Dn(x)
            if self.NA=='NS': Dx=self.Dn[k](x)

            if self.OS=='GD':
                Jr=self.Jaco([x,b])
                tmp=tf.keras.layers.Subtract()([x,Dx]) # residual aliasing and noise-like artifacts
                if self.Lambda_type == 'Diff':
                    tmp=tf.keras.layers.Add()([Jr,self.Lambda[k](tmp)])
                else:
                    tmp=tf.keras.layers.Add()([Jr,self.Lambda(tmp)])
                tmp=tf.keras.layers.Subtract()([x,2*self.Mu[k](tmp)])

            self.Dc['dc'+str(k)]=tmp
        return self.Dc['dc'+str(self.K)]


class DOPAMINE(tf.keras.Model):
    """
    Deep mOdel-based magnetic resonance PArameter MappIng NEtwork `DOPAMINE` (Jun2021).

    Use different `CNN` for different parameter map and Gradient Descent `GD` optimization strategy.
    ### AUGMENTS
    - nLayers: num of convolution layers.
    - K: num of iterations.
    - tes: echo times.
    - TS: Training Strategy. End-to-end Training `ET` or Pre-trained Denoiser `PD`. 
    ### RETURN
    - x: parameter maps.
    """
    def __init__(self,nLayers=5,K=10,tes=None,TS='ET',**kwargs):
        super(DOPAMINE,self).__init__(**kwargs)

        ts_dict = ['ET','PD']
        assert TS in ts_dict, 'Unsupported Training Strategy.'
        if TS == 'ET': trainable=True
        if TS == 'PD': trainable=False

        self.K=K
        self.DRS0   = lay.Denoiser(nLayers=nLayers,out=1,BN=True,trainable=trainable,name='DenoiserS') # parameter map reconstruction network.
        self.DRR2   = lay.Denoiser(nLayers=nLayers,out=1,BN=True,trainable=trainable,name='DenoiserR')
        self.Jaco   = lay.JAT(tes,name='Jr') 
        self.Lambda,self.Mu = {},{}
        for k in range(1,K+1):
            self.Lambda[k] = lay.superPara(initial_value=4.0, trainable=True,name='Lambda'+str(k))
            self.Mu[k]     = lay.superPara(initial_value=0.01,trainable=True,name='Mu'+str(k))

        self.Dc={}

    def call(self,inputs):
        x0 = inputs[0]
        b  = inputs[1]
        self.Dc['dc0']=x0
        for k in range(1,self.K+1):
            x=self.Dc['dc'+str(k-1)]

            S0  = self.DRS0(tf.expand_dims(x[...,0],axis=-1))
            R2  = self.DRR2(tf.expand_dims(x[...,1],axis=-1))
            DRx = tf.keras.layers.Concatenate()([S0,R2])

            Jr  = self.Jaco([x,b])
            delta = tf.keras.layers.Subtract()([x,DRx]) # residual aliasing and noise-like artifacts
            delta = tf.keras.layers.Add()([Jr,self.Lambda[k](delta)])
            x     = tf.keras.layers.Subtract()([x,2*self.Mu[k](delta)])

            self.Dc['dc'+str(k)]=x
        return self.Dc['dc'+str(self.K)]

class Denoiser(tf.keras.Model):
    """
    Only the `Denoiser` block.
    ### AUGMENTS
    - nLayers: num of convolutional layer.
    - Seperated: different `CNN` for different parameter map. `True` or `False`.
    - NA: Network Architechture. The network architecture of the  model using this Denoiser. 
        - When `WS` the trained Denoiser will have a name of `Denoiser`.
        - When `NS` the trained Denoiser will have a name of `Denoisern`.
    - n: is for the name of Denoiser when `NS`.
    ### RETURN
    - x: parameter maps.
    """
    def __init__(self,nLayers=5,Seperated=True,NA='WS',n=1,**kwargs):
        super(Denoiser,self).__init__(**kwargs)
        self.nLayers=nLayers
        
        na_dict = ['WS','NS']

        self.Seperated=Seperated

        assert NA in na_dict, 'Unsupported Network Architecture.'

        if Seperated==True:
            if NA == 'WS':
                self.DnS = lay.Denoiser(nLayers=self.nLayers,out=1,BN=True,trainable=True,name='DenoiserS')
                self.DnR = lay.Denoiser(nLayers=self.nLayers,out=1,BN=True,trainable=True,name='DenoiserR')
            if NA == 'NS':
                self.DnS = lay.Denoiser(nLayers=self.nLayers,out=1,BN=True,trainable=True,name='DenoiserS'+str(n))
                self.DnR = lay.Denoiser(nLayers=self.nLayers,out=1,BN=True,trainable=True,name='DenoiserR'+str(n))
        if Seperated == False:
            if NA == 'WS':
                self.Dn = lay.Denoiser(nLayers=self.nLayers,out=2,BN=True,trainable=True,name='Denoiser')
            if NA == 'NS':
                self.Dn = lay.Denoiser(nLayers=self.nLayers,out=2,BN=True,trainable=True,name='Denoiser'+str(n))
        
    def call(self,inputs):
        x=inputs
        if self.Seperated==True:
            S = self.DnS(tf.expand_dims(x[...,0],axis=-1))
            R = self.DnR(tf.expand_dims(x[...,1],axis=-1))
            x = tf.keras.layers.Concatenate()([S,R])
        if self.Seperated==False:
            x=self.Dn(x)
        return x
