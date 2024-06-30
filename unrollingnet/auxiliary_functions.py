import tensorflow as tf

# def range_constaint_R1(x,A_max=3.0,B_max=6.0,R1_max=50.0):
#     x = tf.math.maximum(x,0.0)
#     A,B,R1 = tf.split(x,3,axis=-1)
#     A = tf.math.minimum(A,A_max)
#     B = tf.math.minimum(B,B_max)
#     R1= tf.math.minimum(R1,R1_max)
#     x = tf.concat([A,B,R1],axis=-1)
#     return x

def range_constaint_R2(x,m0_max=3.0,p2_max=10.0):
    x     = tf.math.maximum(x,0.0)
    m0,p2 = tf.split(x,2,axis=-1)
    m0    = tf.math.minimum(m0,m0_max)
    p2    = tf.math.minimum(p2,p2_max)
    x     = tf.concat([m0,p2],axis=-1)
    return x

class range_constaint_R1(tf.keras.layers.Layer):
    '''T1 relaxation model layer.
    '''
    def __init__(self,A_max=3.0,B_max=6.0,R1_max=50.0,name=None):
        super(range_constaint_R1,self).__init__(name=name)
        self.A_max  = A_max
        self.B_max  = B_max
        self.R1_max = R1_max

    def call(self,x):
        x = tf.math.maximum(x,0.0)
        A,B,R1 = tf.split(x,3,axis=-1)
        A = tf.math.minimum(A,self.A_max)
        B = tf.math.minimum(B,self.B_max)
        R1= tf.math.minimum(R1,self.R1_max)
        x = tf.concat([A,B,R1],axis=-1)
        return x

class physical_model(tf.keras.layers.Layer):
    '''T1 relaxation model layer.
    '''
    def __init__(self,signed=True,name=None):
        super(physical_model,self).__init__(name=name)
        self.signed = signed

    def call(self,x,tau):
        x = tf.math.abs(x)
        A,B,R1 = tf.split(x,3,axis=-1)
        s = tf.math.subtract(A,tf.math.multiply(B,tf.math.exp(tf.math.negative(tf.math.multiply(tau,R1)))))
        if self.signed == False: 
            s = tf.math.abs(s)
        return s