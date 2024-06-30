import tensorflow as tf

def sum_squared_error(y_true, y_pred):
    #return K.mean(K.square(y_pred - y_true), axis=-1)
    #return K.sum(K.square(y_pred - y_true), axis=-1)/2
    return tf.keras.backend.sum(tf.keras.backend.square(y_pred - y_true))/2

def loss_model_exp_image(y_true,y_pred):
    # y_pred: [batch_size, 64, 128,2]
    tes = tf.constant([0.93, 2.27, 3.61, 4.95, 6.29, 7.63, 8.97, 10.4, 11.8, 13.2, 14.6, 16.0])
    # y_pred =y_pred
    y_pred = tf.reshape(y_pred,[-1,y_pred.shape[-1]])
    y_true = tf.reshape(y_true,[-1,12])
    recon  = tf.reshape(y_pred[:,0],[-1,1])*tf.exp(-1.0*tf.reshape(y_pred[:,1],[-1,1])/1000.0*tes)
    # recon = tf.reshape(recon,[-1,32,32,12])
    loss = tf.keras.backend.sum(tf.keras.backend.square(recon - y_true))/2
    return loss