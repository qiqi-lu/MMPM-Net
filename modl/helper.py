import skimage
import glob, re, os
import numpy as np
import matplotlib.pyplot as plt


def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir,'model_*.h5'))  # get name list of all .hdf5 files
    #file_list = os.listdir(save_dir)
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).h5.*",file_)
            #print(result[0])
            epochs_exist.append(int(result[0]))
        initial_epoch=max(epochs_exist)   
    else:
        initial_epoch = 0
    return initial_epoch

def makePairedData(para_maps,tes,sigma=10,noise_type='Guassian'):
    """
    From parameter maps to magnitude images, and add noise.
    ## OUTPUT
    magnitude images, noisy magnitude images.
    """
    if len(para_maps.shape)==3:
        para_maps=para_maps[np.newaxis]
    nte=tes.shape[-1]
    in_shape = para_maps.shape
    s0_vec = np.reshape(para_maps[...,0],[-1,1]) 
    r2_vec = np.reshape(para_maps[...,1],[-1,1])
    s=s0_vec*np.exp(-1*tes/1000.0*r2_vec)
    imgs   = np.reshape(s,[in_shape[-4],in_shape[-3],in_shape[-2],nte]) # may need a batch dim
    imgs_n = addNoise(imgs=imgs,sigma=sigma,noise_type=noise_type)
    return imgs, imgs_n

def addNoise(imgs,sigma=10,noise_type='Gaussian'):
    """
    Add nosie to the inputs.
    ## RETURN
    ### img_n
    Data with noise.
    """
    print('Add '+noise_type+' noise...')
    if noise_type == 'Gaussian':
        imgs_n = imgs+np.random.normal(loc=0,scale=sigma,size=imgs.shape)
    if noise_type == 'Rician':
        r = imgs+np.random.normal(loc=0,scale=sigma,size=imgs.shape)
        i = np.random.normal(loc=0,scale=sigma,size=imgs.shape)
        imgs_n=np.sqrt(r**2+i**2)
    return imgs_n

def makeBlockImage(img_size=(5,5),block_size=(5,5),type='Random',value=None):
    """
    Make a block like image.
    ## INPUT :
    ### type
    Random: value=[low,high], default: [0,1000], random (uniform) value for each block.
    UserDefined: value=[...], user defiened value for each block.
    ## OUTPUT :
    ### image_blcok
    Created block image.
    ### values
    Values correspond to each block, with a shape of [img_size].
    """
    if type=='UserDefined':
        if value!=None & len(value)==img_size[0]*img_size[1]:
            values=value
        else:
            print(':: Error: Inconsistent between img_size and value shape.')
    elif type=='Random':
        if value==None:
            value=[0,1000]
            values=np.random.uniform(value[0],value[1],size=img_size[0]*img_size[1]) # values for each block
        elif len(value)==2:
            values=np.random.uniform(value[0],value[1],size=img_size[0]*img_size[1]) # values for each block
        else:
            print(':: Error: Value should be [low,high].')
    else:
        print(':: Error: Unmanageble Type.')
    
    values=np.reshape(values,img_size)
    img = np.ones([img_size[0],img_size[1],block_size[0],block_size[1]])
    for i in range(0,img_size[0]):
        block_row=img[i,0,:,:]*values[i,0]
        for j in range(1,img_size[1]):
            block_row=np.hstack((block_row,img[i,j,:,:]*values[i,j]))
        if i ==0:
            img_block=block_row
        else:
            img_block=np.vstack((img_block,block_row))
    return img_block, values

def LogLinearN(imgs,TEs,n=3):
    """
    Log-Linear Method.

    Perfrom a pixel-wise linear fit of the decay curve after a log transofrmation (using the first n data point).
    ### AUGMENTS
    - TEs : Echo Time (ms)
    ### RETURN
    - maps : parameter maps [S0, R2*].
    """
    assert len(imgs.shape)==4 or 3, 'Data with shape of [batch,w,h,c] or [w,h,c] is needed.'
    if len(imgs.shape)==3: imgs=imgs[np.newaxis]

    imgs_v=np.reshape(imgs,[-1,imgs.shape[-1]])+1e-7 # images point vector with all data point
    imgs_v=np.abs(imgs_v)
    x=TEs[0:n]/1000.0
    y=np.log(imgs_v[:,0:n]) # logsignal

    x_mean = np.mean(x)
    y_mean = np.reshape(np.mean(y,axis=1),[-1,1])
    w = np.reshape(np.sum((x-x_mean)*(y-y_mean),axis=1)/np.sum((x-x_mean)**2),[-1,1])
    b = y_mean-w*x_mean

    r2_v = -w[:,0]
    s0_v = np.exp(b)[:,0]

    map_v=np.zeros(shape=(imgs_v.shape[0],2))
    map_v[:,0]=np.abs(s0_v)
    map_v[:,1]=np.abs(r2_v)
    maps=np.reshape(map_v,[imgs.shape[0],imgs.shape[1],imgs.shape[2],2])
    return maps

def sum_squared_error(y_true, y_pred):
    import tensorflow as tf
    #return K.mean(K.square(y_pred - y_true), axis=-1)
    #return K.sum(K.square(y_pred - y_true), axis=-1)/2
    return tf.keras.backend.sum(tf.keras.backend.square(y_pred - y_true))/2

def checkNaN(data):
    if len(data.shape)!=2: data=np.reshape(data,[-1,1])
    import pandas as pd
    df=pd.DataFrame(data)
    print(df.isnull().any(axis=0))

def makePatch(data,patch_size=32,stride=8,rescale=True):
    """
    Patching data.
    """
    print('Patching...')
    import cv2

    if len(data.shape)==3:
        data=data[np.newaxis]

    if rescale:
        scales = [1, 0.9, 0.8, 0.7]
    else:
        scales = [1]

    aug_times = 8
    n,h,w,c=data.shape
    patches = []
    for id in range(n):
        for s in scales:
            # rescale image size
            h_scaled, w_scaled = int(h*s),int(w*s)
            img_rescaled = cv2.resize(data[id], (w_scaled,h_scaled), interpolation=cv2.INTER_CUBIC)
            # if len(img_rescaled.shape)==3: img_rescaled = img_rescaled[...,np.newaxis]
            # extract patches
            for i in range(0, h_scaled-patch_size+1, stride):
                for j in range(0, w_scaled-patch_size+1, stride):
                    x = img_rescaled[i:i+patch_size, j:j+patch_size,:]
                    # data aug
                    for k in range(0, aug_times):
                        x_aug = data_aug(x, mode=k)
                        patches.append(x_aug)
    patch = np.array(patches)
    print('Patches shape: ',patch.shape)
    return patch

def data_aug(img, mode=0):
    # aug data size
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))
                    

if __name__ == '__main__':
    import metricx
    tes =np.array([0.93, 2.27, 3.61, 4.95, 6.29, 7.63, 8.97, 10.4, 11.8, 13.2, 14.6, 16.0])
    # tes =np.array([0.80, 1.05, 1.30, 1.55, 1.80, 2.05, 2.30, 2.55, 2.80, 3.05, 3.30, 3.55]) # (Wood2005)3
    sigma=30

    # remake=True
    remake=False
    if remake:
        # Parameter maps
        size=(20,20)
        n=10

        x=np.zeros((n,size[0]*10,size[1]*10,2))
        for i in range(n):
            x[...,0],_=makeBlockImage(img_size=size,block_size=(10,10),type='Random',value=[300,400])
            x[...,1],_=makeBlockImage(img_size=size,block_size=(10,10),type='Random',value=[0,1000])
        print(x.shape)

        # make simulated images
        # imgs,imgs_n,maps=makePairedData(para_maps=x,tes=tes,sigma=sigma,noise_type='Rician')
        imgs,imgs_n=makePairedData(para_maps=x,tes=tes,sigma=sigma,noise_type='Gaussian')
        maps=x
        print(imgs.shape)
        print(imgs_n.shape)
        print(maps.shape)
        
        path = os.path.join('data','test')
        if not os.path.exists(path=path):
            os.makedirs(path)

        np.save(os.path.join(path,'map_'+str(sigma)),maps)
        np.save(os.path.join(path,'imgs_'+str(sigma)),imgs)
        np.save(os.path.join(path,'imgsN_'+str(sigma)),imgs_n)
    
    if remake==False:
        
        maps=np.load(os.path.join('data','test','map_'+str(sigma)+'.npy'))
        imgs=np.load(os.path.join('data','test','imgs_'+str(sigma)+'.npy'))
        imgs_n=np.load(os.path.join('data','test','imgsN_'+str(sigma)+'.npy'))

    maps_n = LogLinearN(imgs_n,tes,n=2)

    plt.figure(figsize=(30,20))
    vmax=400
    i=1
    # images without nosie
    plt.subplot(3,4,1)
    plt.imshow(imgs[i,:,:,0],cmap='gray',vmax=vmax,vmin=0)
    plt.subplot(3,4,2)
    plt.imshow(imgs[i,:,:,1],cmap='gray',vmax=vmax,vmin=0)
    plt.subplot(3,4,3)
    plt.imshow(imgs[i,:,:,2],cmap='gray',vmax=vmax,vmin=0)
    plt.subplot(3,4,4)
    plt.imshow(imgs[i,:,:,3],cmap='gray',vmax=vmax,vmin=0)
    #  images with noise
    plt.subplot(3,4,5)
    plt.imshow(imgs_n[i,:,:,0],cmap='gray',vmax=vmax,vmin=0)
    plt.subplot(3,4,6)
    plt.imshow(imgs_n[i,:,:,1],cmap='gray',vmax=vmax,vmin=0)
    plt.subplot(3,4,7)
    plt.imshow(imgs_n[i,:,:,2],cmap='gray',vmax=vmax,vmin=0)
    plt.subplot(3,4,8)
    plt.imshow(imgs_n[i,:,:,3],cmap='gray',vmax=vmax,vmin=0)
    # parameter maps (reference)
    nRMSEs = metricx.nRMSE(maps[...,0],maps_n[...,0])
    nRMSEr = metricx.nRMSE(maps[...,1],maps_n[...,1])
    print(nRMSEr)
    print(np.mean(nRMSEr))
    SSIMs = metricx.SSIM(maps[...,0],maps_n[...,0],data_range=1024)
    SSIMr = metricx.SSIM(maps[...,1],maps_n[...,1],data_range=1024)
    print(SSIMr)
    print(np.mean(SSIMr))

    plt.subplot(3,4,9)
    plt.imshow(maps[i,:,:,0],cmap='jet',vmax=450,vmin=300),plt.colorbar(fraction=0.024)
    plt.subplot(3,4,10)
    plt.imshow(maps[i,:,:,1],cmap='jet',vmax=1100,vmin=0),plt.colorbar(fraction=0.024)
    plt.subplot(3,4,11)
    plt.imshow(maps_n[i,:,:,0],cmap='jet',vmax=450,vmin=300),plt.colorbar(fraction=0.024),plt.title('RE='+str(nRMSEs[i]),loc='left'),plt.title('SSIM='+str(SSIMs[i]),loc='right')
    plt.subplot(3,4,12)
    plt.imshow(maps_n[i,:,:,1],cmap='jet',vmax=1100,vmin=0),plt.colorbar(fraction=0.024),plt.title('RE='+str(nRMSEr[i]),loc='left'),plt.title('SSIM='+str(SSIMr[i]),loc='right')

    plt.savefig(os.path.join('figures','tmp.png'))

    nRMSEs=[]
    nRMSEr=[]
    SSIMs=[]
    SSIMr=[]
    for i in range(12):
        p = LogLinearN(imgs_n,tes,n=i)
        nRMSEs.append(metricx.nRMSE(maps[...,0],p[...,0],mean=True))
        nRMSEr.append(metricx.nRMSE(maps[...,1],p[...,1],mean=True))
        SSIMs.append(metricx.SSIM(maps[...,0],p[...,0],data_range=1024,mean=True))
        SSIMr.append(metricx.SSIM(maps[...,1],p[...,1],data_range=1024,mean=True))
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1),plt.title('NRMSE')
    plt.plot(np.linspace(1,12,12),nRMSEs,'r',label='S0')
    plt.plot(np.linspace(1,12,12),nRMSEr,'b',label='R2')
    plt.legend()
    plt.subplot(1,2,2),plt.title('SSIM')
    plt.plot(np.linspace(1,12,12),SSIMs,'r',label='S0')
    plt.plot(np.linspace(1,12,12),SSIMr,'b',label='R2')
    plt.legend()
    plt.savefig(os.path.join('figures','plot.png'))
