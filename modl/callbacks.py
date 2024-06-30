# custom callbacks
import tensorflow as tf
class LogSuperPara(tf.keras.callbacks.Callback):
    """
    Log the super parameter every epoch.
    """
    def __init__(self,K=10,Lambda_type=None):
        super(LogSuperPara,self).__init__()
        self.K=K
        self.type=Lambda_type
    
    def on_epoch_begin(self,epoch,logs=None):
        if self.type == 'Same' or type(self.type)==float or type(self.type)==int:
            Lam = self.model.get_layer(name='Lambda').get_weights()[0]
            tf.summary.scalar("Lambda",Lam,step=epoch)
            for i in range(1,self.K+1):
                Mu = self.model.get_layer(name='Mu'+str(i)).get_weights()[0]
                tf.summary.scalar("Mu"+str(i),Mu,step=epoch)
        if self.type == 'Diff':
            for i in range(1,self.K+1):
                Lam = self.model.get_layer(name='Lambda'+str(i)).get_weights()[0]
                Mu  = self.model.get_layer(name='Mu'+str(i)).get_weights()[0]
                tf.summary.scalar('Lambda'+str(i),Lam,step=epoch)
                tf.summary.scalar('Mu'+str(i),Mu,step=epoch)

def lr_schedule(epoch):
    initial_lr = 0.0001
    if   epoch<=500:  lr = initial_lr
    elif epoch<=1000: lr = initial_lr/10
    elif epoch<=1500: lr = initial_lr/20 
    else: lr = initial_lr/20 
    return lr

        
        