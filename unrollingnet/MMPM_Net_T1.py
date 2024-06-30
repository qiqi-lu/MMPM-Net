import tensorflow as tf
from . import denoiser
from . import auxiliary_functions as afunc

class ReconBlock(tf.keras.layers.Layer):
    '''Gaussian-Newton method.
    '''
    def __init__(self,signed=True,name=None):
        super(ReconBlock,self).__init__(name=name)
        self.relax = afunc.physical_model(signed=signed,name='T1_relaxation')
    
    @tf.function
    def jaco(self,x_vec,tau0):
        print('Tracing ...')
        with tf.GradientTape(persistent=True) as g:
            g.watch(x_vec)
            s = self.relax(x_vec,tau0) # [N,Nq]
        ds_dx = g.batch_jacobian(s,x_vec) # [N,Nq,Np]
        return ds_dx

    @tf.function
    def call(self,x,z,beta,rho,sigma,b,tau):
        Nb = tf.shape(x)[0]
        Ny, Nx, Np = tf.shape(x)[-3], tf.shape(x)[-2], tf.shape(x)[-1]
        Nq = tf.shape(tau)[-1]

        ###### Jacobian matrix of physical model ######
        # automatic differentiation
        tau0  = tau[0] # [Nq]
        x_vec = tf.reshape(x,shape=(-1,Np))    # [N,Np]
        ds_dx = self.jaco(x_vec=x_vec,tau0=tau0)
        ds_dx = tf.reshape(ds_dx,shape=(Nb,Ny,Nx,Nq,Np))    # [N,Np]

        ###### Gradient ######
        s = self.relax(x,tau0)        # [Nb,Ny,Nx,Nq]

        r = tf.math.subtract(s,b)[...,tf.newaxis]                   # residual, [Nb,Ny,Nx,Nq,1]
        grad1 = tf.math.reduce_mean(tf.math.multiply(ds_dx,r),axis=-2)          # [Nb,Ny,Nx,Np]
        grad2 = tf.math.multiply(rho,tf.math.subtract(tf.math.add(x,beta),z))   # [Nb,Ny,Nx,Np]
        grad  = tf.math.add(grad1,grad2) # [Nb,Ny,Nx,Np]

        ###### Hessian matrix ######
        # JTJ matrix of physical model
        ds_dx = ds_dx[...,tf.newaxis]                           # [Nb,Ny,Nx,Nq,Np,1]
        JTJ = tf.linalg.matmul(ds_dx,ds_dx,transpose_b=True)    # [Nb,Ny,Nx,Nq,Np,Np]
        # Hessian matrix
        JTJ = tf.math.reduce_sum(JTJ,axis=-3)                   # [Nb,Ny,Nx,Np,Np]
        rho_eye = tf.math.multiply(tf.eye(Np),rho)              # [Np.Np]
        H = tf.math.add(JTJ,rho_eye)                            # Hessian matrix, [Nb,Ny,Nx,Np,Np]

        # Hessian matrix inversion (LM)
        eps = tf.math.multiply(tf.eye(Np),1.e-8)                # [Np,Np]
        idx = tf.where(tf.linalg.det(H) > 0.0, 0.0, 1.0)[..., tf.newaxis, tf.newaxis] # [Nb,Ny,Nx,1,1]
        H   = tf.math.add(tf.math.multiply(idx,eps),H)          # [Nb,Ny,Nx,Np,Np]
        H_inv = tf.linalg.inv(H)                                # [Nb,Ny,Nx,Np,Np]

        grad = grad[...,tf.newaxis] # [Nb,Ny,Nx,Np,1]
        d = tf.linalg.matmul(H_inv,grad)
        d = tf.squeeze(d,axis=-1)
        x = tf.math.subtract(x,d) 
        return x

class AuxVarBlock(tf.keras.layers.Layer):
    """
    Auxiliary variable update block based on fixed-point method.
    In  : x,z,beta,rho,lam,fz
    Out : z   
    """
    def __init__(self,name=None):
        super(AuxVarBlock,self).__init__(name=name)

    def call(self,x,z,beta,rho,lam,fz):
        z_1 = tf.math.multiply(rho,tf.math.add(x,beta))
        z_2 = tf.math.multiply(lam,fz)
        z_3 = tf.math.divide_no_nan(tf.math.add(z_1,z_2),tf.math.add(rho,lam))
        return z_3

class MMPM_Net(tf.keras.layers.Layer):
    def __init__(self,Ns=10,Nk=1,Nt=1,Np=3,signed=True,f=3,sep=False,ini_lam=0.001,ini_rho=0.1,test_mode=False,name=None):
        super(MMPM_Net,self).__init__(name=name)
        self.Ns = Ns # number of stages
        self.Nk = Nk # number of iterations in reconstruction block
        self.Nt = Nt # number of iterations in auxiliary variable update block

        self.lam   = tf.Variable(initial_value=ini_lam,trainable=True,name=name+'_lam',constraint=tf.keras.constraints.NonNeg())
        self.rho   = tf.Variable(initial_value=ini_rho,trainable=True,name=name+'_rho',constraint=tf.keras.constraints.NonNeg())
        self.sigma = tf.Variable(initial_value=1.0,trainable=False,name=name+'_sigma', constraint=tf.keras.constraints.NonNeg()) # useless now.

        self.reconblock  = ReconBlock(signed=signed,name='recon')
        self.auxvarblock = AuxVarBlock(name='auxvar')
        self.range_cons  = afunc.range_constaint_R1(A_max=3.0,B_max=6.0,R1_max=50.0,name='range_cons')

        self.tmode = test_mode
        self.sep = sep # whether to process each channel seperately

        if self.sep == True:
            self.denoiser = []
            for i in range(Np):
                self.denoiser.append(denoiser.CNN(Nconv=7,Np=1,filters=64,f=f,name='denoiser_'+str(i)))

        if self.sep == False:
                self.denoiser = denoiser.CNN(Nconv=7,Np=Np,filters=64,f=f,name='denoiser')
    
    def call(self,inputs):
        b   = inputs[0]
        tau = inputs[1]
        xm  = []
        
        ###### INITIALIZATION ######
        A  = tf.math.reduce_max(tf.math.abs(b),axis=-1)
        B  = tf.math.multiply(2.0,A)
        R1 = tf.math.multiply(tf.ones_like(A),1.0)
        x  = tf.stack([A,B,R1],axis=-1)
        x  = self.range_cons(x)
        z  = x
        beta = tf.zeros_like(x)

        if self.tmode == True: xm.append(x) # test
        ##########################
        for _ in range(self.Nk):
            x = self.reconblock(x,z,beta,0.0,self.sigma,b,tau) 
            x = self.range_cons(x)
        z = x
        beta = tf.zeros_like(x)

        if self.tmode == True: xm.append(x) # test
        
        ###### Main Iterations ######
        for _ in range(self.Ns):
            # Reconstruction
            for _ in range(self.Nk):
                x = self.reconblock(x,z,beta,self.rho,self.sigma,b,tau) 
                x = self.range_cons(x)
            xm.append(x) # train & test
            # Auxiliary variable update 
            for _ in range(self.Nt):
                if self.sep == False:
                    fz = self.denoiser(z)
                    
                if self.sep == True:
                    z0,z1,z2 = tf.split(z,3,axis=-1)
                    z0 = self.denoiser[0](z0)
                    z1 = self.denoiser[1](z1)
                    z2 = self.denoiser[2](z2)
                    fz = tf.keras.layers.Concatenate(axis=-1)([z0,z1,z2])

                z = self.auxvarblock(x,z,beta,self.rho,self.lam,fz)
                z = self.range_cons(z)
            if self.tmode == False: xm.append(fz) # train
            if self.tmode == False: xm.append(z)  # train
            # Multipliers update
            beta = tf.math.subtract(tf.math.add(beta,x),z)

        if self.tmode == False: xm.append(x) # train
        return tf.stack(xm)
