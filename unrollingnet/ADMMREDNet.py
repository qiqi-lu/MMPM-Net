import tensorflow as tf
from . import denoiser

class ReconBlock(tf.keras.layers.Layer):
    """
    Gaussian-Newton method.
    In  : x,z,beta,rho,sigma,b,tes
    Out : x
    """
    def __init__(self,name=None):
        super(ReconBlock,self).__init__(name=name)

    def call(self,x,z,beta,rho,sigma,b,tes):
        m0,p2   = tf.split(x,2,axis=-1)

        tes     = tes[:,tf.newaxis,tf.newaxis,:]
        exp     = tf.math.exp(tf.math.negative(tf.math.multiply(tes,p2)))

        Ax      = tf.math.multiply(m0,exp)
        r       = tf.math.subtract(Ax,b) # residual

        dSdm0   = exp
        dSdp2   = tf.math.multiply(tf.math.negative(tes),tf.math.multiply(m0,dSdm0))

        Jr_m0   = tf.math.reduce_mean(tf.math.multiply(dSdm0,r),axis=-1)
        Jr_p2   = tf.math.reduce_mean(tf.math.multiply(dSdp2,r),axis=-1)
        Jr      = tf.stack([Jr_m0,Jr_p2],axis=-1)

        w       = tf.math.divide_no_nan(1.0,tf.math.square(sigma))

        grad    = tf.math.subtract(tf.math.add(x,beta),z)
        grad    = tf.math.add(tf.math.multiply(w,Jr),tf.math.multiply(rho,grad)) # gradient


        dfdm02  = tf.math.square(dSdm0)
        dfdm02  = tf.math.add(tf.math.multiply(w,tf.math.reduce_mean(dfdm02,axis=-1)),rho)

        dfdp22  = tf.math.square(dSdp2)
        dfdp22  = tf.math.add(tf.math.multiply(w,tf.math.reduce_mean(dfdp22,axis=-1)),rho)

        dfdm0p2 = tf.math.multiply(dSdm0,dSdp2)
        dfdm0p2 = tf.math.multiply(w,tf.math.reduce_mean(dfdm0p2,axis=-1))

        ### cofactor-det inversion
        c1       = tf.stack([dfdp22,dfdm0p2],axis=-1)
        c2       = tf.stack([dfdm0p2,dfdm02],axis=-1)
        cofactor = tf.stack([c1,c2],axis=-2)
        cofactor = tf.math.multiply(cofactor,[[1.0,-1.0],[-1.0,1.0]])
        det      = tf.math.subtract(tf.math.multiply(dfdm02,dfdp22),tf.math.square(dfdm0p2))[...,tf.newaxis,tf.newaxis]
        det      = tf.where(det<0.0,0.0,det)
        hessian_inv = tf.math.divide_no_nan(cofactor,det)

        #### LU decomposition inversion
        # eps     = 1.e-8*tf.eye(2)
        # c1      = tf.stack([dfdm02,dfdm0p2],axis=-1)
        # c2      = tf.stack([dfdm0p2,dfdp22],axis=-1)
        # hessian = tf.stack([c1,c2],axis=-2)
        # idx     = tf.where(tf.linalg.det(hessian) > 0.0, 0.0, 1.0)
        # hessian = idx[..., tf.newaxis, tf.newaxis]*eps + hessian
        # hessian_inv = tf.linalg.inv(hessian)

        grad    = grad[...,tf.newaxis,:]
        d       = tf.math.multiply(hessian_inv,grad)
        d       = tf.math.reduce_sum(d,axis=-1)
        x       = tf.math.subtract(x,d) 
        return x

class AuxVarBlock(tf.keras.layers.Layer):
    """
    Fixed-point method.
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

def range_constaint(x,m0_max=3.0,p2_max=10.0):
    x     = tf.math.maximum(x,0.0)
    m0,p2 = tf.split(x,2,axis=-1)
    m0    = tf.math.minimum(m0,m0_max)
    p2    = tf.math.minimum(p2,p2_max)
    x     = tf.concat([m0,p2],axis=-1)
    return x

class ADMMNetm(tf.keras.layers.Layer):
    def __init__(self,Ns=10,Nk=1,Nt=1,f=3,path=1,name=None):
        super(ADMMNetm,self).__init__(name=name)
        self.Ns = Ns
        self.Nk = Nk
        self.Nt = Nt
        self.lam   = tf.Variable(initial_value=0.001,trainable=True,name=name+'_lam',constraint=tf.keras.constraints.NonNeg())
        self.rho   = tf.Variable(initial_value=0.1,trainable=True,name=name+'_rho',constraint=tf.keras.constraints.NonNeg())
        self.sigma = tf.Variable(initial_value=1.0,trainable=False,name=name+'_sigma',constraint=tf.keras.constraints.NonNeg())
        self.reconblock  = ReconBlock(name='recon')
        self.auxvarblock = AuxVarBlock(name='auxvar')
        self.path = path
        if path == 2: Np=1
        if path == 1: Np=2
        self.denoiser = []
        for i in range(self.path):
            self.denoiser.append(denoiser.CNN(Nconv=7,Np=Np,filters=64,f=f,name='denoiser_'+str(i)))
            # self.denoiser.append(denoiser.CNN_Dilate(Nconv=7,Np=Np,filters=64,f=f,norm=False,name='denoiser_'+str(i)))
            # self.denoiser.append(denoiser.DnCNN(Nconv=20,Np=Np,filters=64,f=f,norm=False,name='denoiser_'+str(i)))
            # self.denoiser.append(denoiser.DnCNN(Nconv=20,Np=Np,filters=64,f=f,norm=True,name='denoiser_'+str(i)))
    
    def call(self,inputs):
        b   = inputs[0]
        tes = inputs[1]
        
        ##### INITIALIZATION #####
        m0 = tf.math.reduce_max(b,axis=-1)
        p2 = tf.math.add(tf.math.multiply(b[...,0],0.0),1.0)
        x  = tf.stack([m0,p2],axis=-1)
        x = range_constaint(x,m0_max=3.0,p2_max=10.0)

        z = x
        beta = tf.math.multiply(b[...,0:2],0.0)

        xm   = []
        xm.append(x) # test
        for _ in range(self.Nk):
            x = self.reconblock(x,z,beta,0.0,self.sigma,b,tes) 
            x = range_constaint(x,m0_max=3.0,p2_max=10.0)
        ## xm.append(x)
        z = x
        beta = tf.math.multiply(b[...,0:2],0.0)

        for _ in range(self.Ns):
            for _ in range(self.Nk):
                x = self.reconblock(x,z,beta,self.rho,self.sigma,b,tes) 
                x = range_constaint(x,m0_max=3.0,p2_max=10.0)
            xm.append(x) # train & test
            
            for _ in range(self.Nt):
                if self.path == 1:
                    fz = self.denoiser[0](z)
                    
                if self.path == 2:
                    z0,z1 = tf.split(z,2,axis=-1)
                    z0 = self.denoiser[0](z0)
                    z1 = self.denoiser[1](z1)
                    fz = tf.keras.layers.Concatenate(axis=-1)([z0,z1])

                z = self.auxvarblock(x,z,beta,self.rho,self.lam,fz)
                z = range_constaint(z,m0_max=3.0,p2_max=10.0)
            # xm.append(fz) # train
            # xm.append(z)  # train

            beta = tf.math.subtract(tf.math.add(beta,x),z)

        # xm.append(x) # train
        return tf.stack(xm)
