import numpy as np
import matplotlib.pyplot as plt
import gzip
import struct
import tqdm
import scipy
from scipy.optimize import least_squares
from scipy.special import ive
from scipy import optimize
import cv2
import tensorflow as tf
import os
import glob
import re
import skimage.metrics as metric
import skimage.restoration as restoration
import time
import multiprocessing
import operator
import pydicom
import h5py
import SimpleITK as sitk

def T1_to_T1s(T1,FA,TR):
    'Convert T1 into T1* according to Look-Locker sequence.'
    np.seterr(divide='ignore',invalid='ignore')
    T1s = np.divide(1.0,1.0/T1-np.log(np.cos(FA/360*2*np.pi))/TR,out=np.zeros_like(T1),where=T1!=0)
    return T1s

def rescale_wimg(img,percent=99):
    scale = np.percentile(img[...,0],percent)
    img_rescale = img/scale
    return(img_rescale)

def resize_wimg(img,scale_factor=2.0):
    Ndim = len(img.shape)
    if Ndim == 3:
        Nc = img.shape[-1]
        img_resize = []
        for i in range(Nc):
            im = cv2.resize(img[...,i],None,fx=scale_factor,fy=scale_factor,interpolation=cv2.INTER_LINEAR)
            img_resize.append(im)
        img_resize = np.stack(img_resize)
        img_resize = np.transpose(img_resize,axes=(1,2,0))
    if Ndim == 2:
        img_resize = cv2.resize(img,None,fx=scale_factor,fy=scale_factor,interpolation=cv2.INTER_LINEAR)
    return img_resize

def read_dicom(file_dir,suffix='dcm'):
    file_dir    = os.path.join(file_dir,'*.'+suffix)
    file_names  = sorted(glob.glob(file_dir),key=os.path.getmtime,reverse=True)
    num_files   = np.array(file_names).shape[0]
    imgs,tes    = [],[]
    for file,_ in zip(file_names, range(0,num_files)):
        d = pydicom.dcmread(file)
        imgs.append(d.pixel_array)
        try:    tes.append(d.EchoTime)
        except: tes.append(d.PerFrameFunctionalGroupsSequence[0].MREchoSequence[0].EffectiveEchoTime)
    imgs = [x for i,x in sorted(zip(tes,imgs))]
    tes  = sorted(tes)
    imgs = np.array(imgs).transpose((1,2,0))
    return imgs,np.array(tes)

def read_mat(file_dir):
    with h5py.File(file_dir) as f:
        data = {}
        for k,v in f.items(): data[k] = np.array(v)
        print(data.keys())
    return data

def closing_hole(x):
    '''
    Close small hole in roi.
    - x, [N,Ny,Nx,1].
    '''
    N = x.shape[0]
    y = np.zeros_like(x)
    for i in range(N):
        tmp = x[i,...,0].astype(bool)
        tmp = scipy.ndimage.binary_fill_holes(tmp)
        tmp = np.logical_not(tmp)
        tmp = scipy.ndimage.binary_closing(tmp,border_value=1)
        tmp = np.logical_not(tmp)
        y[i] = tmp[...,np.newaxis].astype(int)
    return y

def reshape_realization(X,Nslice,Nrealization):
    '''
    [Nslice*Nrealization,Ny,Nx] -> [Nrealization,Nslice,Ny,Nx]
    - X, parameter maps, [Nslice*Nrealization,Ny,Nx]
    '''
    _,Ny,Nx   = X.shape
    X_reshape = np.reshape(X,newshape=(Nslice,Nrealization,Ny,Nx))
    X_reshape = np.transpose(X_reshape,axes=(1,0,2,3))
    return X_reshape

def relative_bias(Y,X):
    Y_bar = np.mean(Y,axis=0)
    # rb    = np.where(X[0]>0.0,(Y_bar-X[0])/X[0]*100.0,0.0)
    rb = np.divide(Y_bar-X[0],X[0],out=np.zeros_like(X[0]),where=X[0]!=0)
    rb = rb*100.0
    return rb

def repeatability_coefficient(Y):
    Y_bar   = np.mean(Y,axis=0)
    Y_sigma = np.std(Y,axis=0)
    rc = np.divide(Y_sigma,Y_bar,out=np.zeros_like(Y_sigma),where=Y_bar!=0)
    rc = 2.77*rc*100.0
    return rc

def relative_bias_map(Y,X,Nslice,Nrealization):
    Y_reshape = reshape_realization(Y,Nslice,Nrealization)
    X_reshape = reshape_realization(X,Nslice,Nrealization)
    map_rb    = relative_bias(Y_reshape,X_reshape)
    return map_rb

def repeatability_coefficient_map(Y,Nslice,Nrealization):
    Y_reshape = reshape_realization(Y,Nslice,Nrealization)
    map_rc    = repeatability_coefficient(Y_reshape)
    return map_rc

def read_binary(file_name,data_type='uint8',shape=None):
    """
    Read suppressed binary file to image with shape of `shape`.
    ### ARGUMENT
    - file_name, raw data file file name.
    - data_type, data type.
    - shape, image shape.
    
    ### RETURN
    - data, image data with shape of `shape`.
    """
    dic = {'uint8':'B','int8':'b'}
    assert data_type in list(dic.keys()), 'Unsupported data type. Only for '+str(list(dic.keys()))
    size = 1
    for i in range(len(shape)): size=size*shape[i]
    suffix = os.path.splitext(file_name)[-1]
    if suffix == '.gz': 
        with gzip.open(file_name,'rb') as file: data = file.read()
    if suffix == '.raw': 
        with open(file_name,'rb') as file: data = file.read()
    data = struct.unpack(dic[data_type]*size,data)
    data = np.reshape(data,shape)
    return data

def Relax_T2(A,T2,tau):
    """
    mono-exponential function, S = S0*exp(-te/T2).
    """
    s  = []
    Nq = tau.shape[-1]
    for i in range(Nq):
        sq = np.where(T2<=0.0,0.0,A*np.exp(-tau[i]/T2))
        s.append(sq)
    s = np.stack(s)
    s = np.moveaxis(s,source=0,destination=-1)
    return s

def Relax_T1_three_para_magn(A,B,T1,tau,signed=True):
    """
    mono-exponential function, s = A-B*exp(-tau/T1).
    - A,B,T1, [Ny,Nx]
    - tau, [Nq]
    """
    A,B,T1 = A[...,np.newaxis],B[...,np.newaxis],T1[...,np.newaxis]
    if signed == True : sq = np.where(T1<=0.0,0.0,A-B*np.exp(-tau/T1))
    if signed == False: sq = np.where(T1<=0.0,0.0,np.abs(A-B*np.exp(-tau/T1)))
    return sq

def patch(data,mask,patch_size=40,step_size=20):
    """ 
    Generate patches.
    #### ARGUMENTS
    - data, [Nz,Ny,Nx,Nc]
    - mask, [Nz,Ny,Nx]

    #### RETURN
    - patches, [N,Ny,Nx,Nc].
    """
    Nz,Ny,Nx,_ = data.shape
    num_step_x = (Nx-patch_size)//step_size
    num_step_y = (Ny-patch_size)//step_size
    patches    = []
    pbar=tqdm.tqdm(desc='patching',total=(num_step_x+1)*(num_step_y+1)*Nz)
    for k in range(Nz):
        for i in range(num_step_y+1):
            for j in range(num_step_x+1):
                pbar.update(1)
                if mask[k,i*step_size+int(patch_size/2),j*step_size+int(patch_size/2)] > 0:
                    patches.append(data[k,i*step_size:i*step_size+patch_size,j*step_size:j*step_size+patch_size,:])
    pbar.close()
    patches=np.stack(patches)
    print(patches.shape)
    return patches

def addNoise(imgs,sigma,noise_type='Gaussian',NCoils=1):
    """
    Add nosie to the inputs with fixed standard deviation.
    ### ARGUMENTS
    - imgs, the image to be added noise.
    - sigma, noise sigma.
    - noise_type, type of noise.

    ### RETURN
    - imgs_n: images with noise.
    """
    assert sigma  > 0.0, 'Noise sigma must higher than 0.'
    assert noise_type in ['Rician', 'Gaussian','ChiSquare'], 'Unsupported noise type.'
    assert NCoils > 0.0, 'Coils number should larger than 0.'
    bound = 4.0
    if noise_type == 'Gaussian':
        noise   = np.random.normal(loc=0.0,scale=sigma,size=imgs.shape)
        noise   = np.maximum(np.minimum(noise,bound*sigma),-1.0*bound*sigma)
        imgs_n  = imgs + noise

    if noise_type == 'Rician':
        noise_r = np.random.normal(loc=0.0,scale=sigma,size=imgs.shape)
        noise_r = np.maximum(np.minimum(noise_r,bound*sigma),-1.0*bound*sigma)
        r       = imgs + noise_r
        noise_i = np.random.normal(loc=0.0,scale=sigma,size=imgs.shape)
        noise_i = np.maximum(np.minimum(noise_i,bound*sigma),-1.0*bound*sigma)
        imgs_n  = np.sqrt(r**2 + noise_i**2)

    if noise_type == 'ChiSquare':
        imgs = imgs/np.sqrt(NCoils)
        imgs_n = np.zeros(imgs.shape)
        for _ in range(NCoils):
            noise_r = np.random.normal(loc=0.0,scale=sigma,size=imgs.shape)
            noise_r = np.maximum(np.minimum(noise_r,bound*sigma),-1.0*bound*sigma)
            r       = imgs+noise_r
            noise_i = np.random.normal(loc=0.0,scale=sigma,size=imgs.shape)
            noise_i = np.maximum(np.minimum(noise_i,bound*sigma),-1.0*bound*sigma)
            imgs_n  = imgs_n + r**2 + noise_i**2
        imgs_n = np.sqrt(imgs_n)
    return imgs_n

def addNoiseMix(imgs,sigma_low,sigma_high,noise_type='Gaussian',NCoils=1):
    """
    Add nosie to the inputs with mixed standard deviation.
    ### ARGUMENTS
    - imgs, [N,Nx,Ny,Nc].
    - sigma_low, sigma low bounding.
    - sigma_high, sigma high bounding.
    - noise_type, type of noise.

    ### RETURN
    - imgs_n: images with noise.
    """
    N = imgs.shape[0]
    imgs_n = np.zeros_like(imgs)
    RanGen = np.random.default_rng(seed=0)
    sigmas = RanGen.uniform(low=sigma_low,high=sigma_high,size=N)
    pbar   = tqdm.tqdm(total=N,desc='Add Noise (mix)')
    for i in range(N):
        pbar.update(1)
        imgs_n[i] = addNoise(imgs[i],sigma=sigmas[i],noise_type=noise_type,NCoils=NCoils)
    pbar.close()
    return imgs_n

def model_exp_mono_t(c,x,y):
    return y - c[0]*np.exp(-x/c[1])

def model_exp_mono_r(c,x,y):
    return y - c[0]*np.exp(-x*c[1])

def model_T1_three_para_magn_signed_t(c,x,y):
    return y - (c[0]-c[1]*np.exp(-x/c[2]))

def model_T1_three_para_magn_signed_r(c,x,y):
    return y - (c[0]-c[1]*np.exp(-x*c[2]))

def model_T1_three_para_magn_unsigned_t(c,x,y):
    return y - np.abs(c[0]-c[1]*np.exp(-x/c[2]))

def model_T1_three_para_magn_unsigned_r(c,x,y):
    return y - np.abs(c[0]-c[1]*np.exp(-x*c[2]))


def LogLinear(s,tau,n=0,parameter_type='time'):
    if n == 0: n = s.shape[-1]
    x = tau[0:n]
    y = np.maximum(s,0.0)
    y = np.log(y[...,0:n]+1e-6) # log signal

    x_mean = np.mean(x,axis=-1,keepdims=True)
    y_mean = np.mean(y,axis=-1,keepdims=True)
    w = np.sum((x-x_mean)*(y-y_mean),axis=-1,keepdims=True)/np.sum((x-x_mean)**2,axis=-1,keepdims=True)
    c = y_mean-w*x_mean
    if parameter_type == 'time':
        w  = np.where(w == 0.0, 1e-7, w)
        p1 = np.where(s[...,0] <= 0.0, 0.0, -1.0/w[...,0])
    if parameter_type == 'rate':
        p1 = np.where(s[...,0] <= 0.0, 0.0, -1.0*w[...,0])
    p0 = np.where(s[...,0] <= 0.0, 0.0, np.exp(c[...,0]))
    p = np.stack((p0,p1),axis=-1)
    return p

def exp_mono(s,tau,parameter_type='time',**kwargs):
    '''s = p0*exp(-tau/p1) / s = p0*exp(-tau*p1).
    '''
    s_max = np.max(np.abs(s))
    try:
        if parameter_type == 'time':
            if s[0] <= 0.0: p0, p1 = 0.0, 0.0
            if s[0] >  0.0:
                x0 = np.array([s_max, 0.5*tau[-1]]) 
                bounds = ([0.0, 0.1], [2.0*s_max, 3000.0])
                res = least_squares(model_exp_mono_t, x0, args=(tau, s), bounds = bounds)
                p0, p1 = res.x[0], res.x[1]
        if parameter_type == 'rate':
            if s[0] <= 0.0: p0, p1 = 0.0, 0.0
            if s[0] >  0.0:
                x0 = np.array([s_max, 2.0/tau[-1]]) 
                bounds = ([0.0, 0.00025], [2.0*s_max, 10.0])
                res = least_squares(model_exp_mono_r, x0, args=(tau, s), bounds = bounds)
                p0, p1 = res.x[0], res.x[1]
    except:
        print('Warnning: fitting error.')
        p = LogLinear(s,tau=tau,parameter_type=parameter_type)
        p0,p1 = p[0],p[1]
    return [p0, p1]

def fitting_T1_three_para_magn(s,tau,parameter_type='time',signed=True,**kwargs):
    '''
    T1 mapping, three parameter model, `s=p0-p1*exp(-tau/p2)` or `s=p0-p1*exp(-tau*p2)`.
    - s, signal. [Nq]
    - tau, TE or TI (ms). [Nq]
    - parameter_type, `time` or `rate`.
    '''
    s_max = np.max(np.abs(s))
    try:
        if parameter_type == 'time':
            x0 = np.array([s_max, 2.0*s_max, 1000.0]) 
            bounds = ([0.0, 0.0, 0.1], [np.inf, np.inf, 10000.0])
            if signed == True : res = least_squares(model_T1_three_para_magn_signed_t, x0, args=(tau, s), bounds = bounds)
            if signed == False: res = least_squares(model_T1_three_para_magn_unsigned_t, x0, args=(tau, s), bounds = bounds)
            p0, p1, p2 = res.x[0], res.x[1], res.x[2]
        if parameter_type == 'rate':
            x0 = np.array([s_max, 2.0*s_max, 1.0/1000.0]) 
            bounds = ([0.0, 0.0, 0.0001], [np.inf, np.inf, 10.0])
            if signed == True : res = least_squares(model_T1_three_para_magn_signed_r, x0, args=(tau, s), bounds = bounds)
            if signed == False: res = least_squares(model_T1_three_para_magn_unsigned_r, x0, args=(tau, s), bounds = bounds)
            p0, p1, p2 = res.x[0], res.x[1], res.x[2]
    except:
        print('Warnning: fitting error. Set parameters as zero.')
        p0, p1, p2 = 0, 0, 0 
    return [p0, p1, p2]

def invalid_model(x):
    raise Exception('Invalid Model.')

def PixelWiseMapping(imgs,tau,fitting_model='exp_mono',signed=True,parameter_type='time',sigma=None,NCoils=1,pbar_disable=False,pbar_leave=True,ac=1,**kwargs):
    models = {'exp_mono': exp_mono,
              'T1_three_para_magn':fitting_T1_three_para_magn,
              }
    fun = models.get(fitting_model,invalid_model)
    Ny,Nx,Nc = imgs.shape
    maps = []
    imgs = np.reshape(imgs,(-1,Nc))

    global func_fun
    def func_fun(i):
        para = fun(s=imgs[i],tau=tau,parameter_type=parameter_type,signed=signed,sigma=sigma,NCoils=NCoils)
        return (i, para)

    # parallel computation
    pool = multiprocessing.Pool(ac)

    pbar = tqdm.tqdm(total=Ny*Nx,desc='Pixel-wise Fitting',leave=pbar_leave,disable=pbar_disable)
    for kp in pool.imap_unordered(func_fun,range(Ny*Nx)):
        maps.append(kp)
        pbar.update(1)
    pool.close()
    pbar.close()

    # sort pixel and arrange to original position.
    maps = sorted(maps,key=operator.itemgetter(0))
    maps = [pixel[1] for pixel in maps]
    maps = np.reshape(np.stack(maps),(Ny,Nx,-1))
    return maps

def PCANR(imgs,tes,h,f=5,m=0,Ncoils=1,pbar_leave=True):
    """
    Pixelwise curve-fitting with adaptive neighbor regularization `PCANR` methods for R2* parameter reconstruction.
    (Only for Rician noise)

    ### AUGMENTS
    - imgs: input measured images. [Ny,Nx,Nc]
    - tes: echo times.
    - sigma: background noise sigma.
    - f: neighbour size (2f+1)*(2f+1).
    - m: similarity patch size.
    - beta: regularization parameter.
    - Ncoils: (unused), when apply to multi coils, the noise correction model need to be modified.
    - pbar_leave: whether to leave pbar after ending.

    ### RETURN
    - maps: parameter maps. [h,w,[S0 T2]]
    """
    assert len(imgs.shape) == 3, 'Unsupported size, [Ny,Nx,Nc] is needed.'
    row,col,c = imgs.shape

    imgs = np.pad(imgs,pad_width=((f+m,f+m),(f+m,f+m),(0,0)),mode='symmetric')
    S0   = np.zeros(shape=(row+f+m,col+f+m))
    T2   = np.zeros(shape=(row+f+m,col+f+m))
    # one  = np.ones(shape=((2*f+1)**2,1))

    time.sleep(1.0)
    pbar = tqdm.tqdm(total=row*col,desc='PCANR',leave=pbar_leave)

    for i in range(f+m,f+m+row):
        for j in range(f+m,f+m+col):
            pbar.update(1)
            win = np.zeros((2*f+1,2*f+1)) # weight matrix
            for k in range(i-f,i+f+1):
                for l in range(j-f,j+f+1):
                    win[k-i+f,l-j+f] = (np.linalg.norm(imgs[i-m:i+m+1,j-m:j+m+1,:]-imgs[k-m:k+m+1,l-m:l+m+1,:]))**2 # distance calculation
            win = np.exp(-win/(h**2)) # average on each pixel
            win = win/np.sum(win) # normalization
            # win = np.reshape(win,((2*f+1)**2,1))

            p = imgs[i-f:i+f+1,j-f:j+f+1,:] # data point in searching window
            # p = np.reshape(p,((2*f+1)**2,c))

            S0_0 = np.max(np.abs(imgs[i,j,:]))
            T2_0 = 0.5*tes[-1]
            c0   = np.array([S0_0, T2_0]) 
            bounds = ((0.0, 10.0*S0_0+1.0), (0.1, 3000.0))
            def costfun(c,x,y):
                ##### first-moment noise correction mdeol #####
                # s = c[0]*np.exp(-x*c[1])
                # alpha = (0.5*s/sigma)**2
                # tempw = (1+2*alpha)*ive(0,alpha)+2*alpha*ive(1,alpha)
                # fun_ncexp = np.sqrt(0.5*np.pi*sigma**2)*tempw
                # cost  = np.sum(np.sum((win*(y-one*fun_ncexp))**2,1))

                ##### second-moment noise correction model #####
                # fun_ncexp = c[0]**2*np.exp(-2*x*c[1])+2*sigma**2
                # cost  = np.sum(np.sum((win*(y-one*np.sqrt(fun_ncexp)))**2,1))

                ##### mono-exponential model #####
                fun_exp = c[0]*np.exp(-x/c[1])
                # cost    = np.sum((win*(y-one*fun_exp))**2)
                cost    = 0.5*np.sum(win[...,np.newaxis]*(y-fun_exp)**2)
                return cost

            res = optimize.minimize(costfun, c0, args=(tes, p),bounds=bounds)
            S0[i,j],T2[i,j] = res.x[0], res.x[1]
    pbar.close()
    map = np.stack([S0,T2],axis=-1)
    return map[f+m:,f+m:,:]

# The following function can be used to convert a value to a type comptible with tf.train.Example.
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def write2TFRecord_noise_free(img_gt,map_gt,seg,tes,filename):
    """
    Write numpy array to a TFRecord file.
    ### ARGUMENTS
    - img_gt, [N,Nx,Ny,Nc].
    - map,    [N,Nx,Ny,Np].
    - seg,    [N,Nx,Ny,1].
    - tes,    [N,Nc]
    - filename, `filename`.tfrecords.
    """
    filename = filename+'.tfrecords'
    print('Write to ',filename)
    writer = tf.io.TFRecordWriter(filename) # writer taht will store data to disk
    N,Nx,Ny,Nc = img_gt.shape
    _,_,_,Np   = map_gt.shape

    for i in range(N):
        feature = {
            'img_gt': _bytes_feature(tf.io.serialize_tensor(img_gt[i])),
            'map_gt': _bytes_feature(tf.io.serialize_tensor(map_gt[i])),
            'seg':    _bytes_feature(tf.io.serialize_tensor(seg[i])),
            'tes':    _bytes_feature(tf.io.serialize_tensor(tes[i])),
            'Nx': _int64_feature(Nx),
            'Ny': _int64_feature(Ny),
            'Nc': _int64_feature(Nc),
            'Np': _int64_feature(Np),
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        example = example.SerializeToString()
        writer.write(example)
    writer.close()
    print('> Wrote '+str(N)+' elemets to TFRecord')

def write2TFRecord_noise(imgs_gt_bi,imgs_n_bi,imgs_gt_mono,imgs_n_mono,maps_gt,seg,tes,filename):
    print('Write to ',filename)
    writer = tf.io.TFRecordWriter(filename) # writer taht will store data to disk
    N,Nx,Ny,Nc = imgs_gt_bi.shape
    _,_,_,Np   = maps_gt.shape

    for i in range(N):
        feature = {
            'imgs_gt_bi':   _bytes_feature(tf.io.serialize_tensor(imgs_gt_bi[i])),
            'imgs_n_bi':    _bytes_feature(tf.io.serialize_tensor(imgs_n_bi[i])),
            'imgs_gt_mono': _bytes_feature(tf.io.serialize_tensor(imgs_gt_mono[i])),
            'imgs_n_mono':  _bytes_feature(tf.io.serialize_tensor(imgs_n_mono[i])),
            'maps_gt':  _bytes_feature(tf.io.serialize_tensor(maps_gt[i])),
            'seg':      _bytes_feature(tf.io.serialize_tensor(seg[i])),
            'tes':      _bytes_feature(tf.io.serialize_tensor(tes[i])),
            'Nx': _int64_feature(Nx),
            'Ny': _int64_feature(Ny),
            'Nc': _int64_feature(Nc),
            'Np': _int64_feature(Np),
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        example = example.SerializeToString()
        writer.write(example)
    writer.close()
    print('> Wrote '+str(N)+' elemets to TFRecord')

def parse_noise_free(example):
    feature_discription={
        'img_gt': tf.io.FixedLenFeature([], tf.string),
        'map_gt': tf.io.FixedLenFeature([], tf.string),
        'seg':    tf.io.FixedLenFeature([], tf.string),
        'tes':    tf.io.FixedLenFeature([], tf.string),
        'Nx': tf.io.FixedLenFeature([], tf.int64),
        'Ny': tf.io.FixedLenFeature([], tf.int64),
        'Nc': tf.io.FixedLenFeature([], tf.int64),
        'Np': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed_example = tf.io.parse_single_example(example,feature_discription)

    Nx = parsed_example['Nx']
    Ny = parsed_example['Ny']
    Nc = parsed_example['Nc']
    Np = parsed_example['Np']

    img_gt  = tf.io.parse_tensor(parsed_example['img_gt'], out_type = tf.float32)
    map_gt  = tf.io.parse_tensor(parsed_example['map_gt'], out_type = tf.float32)
    seg     = tf.io.parse_tensor(parsed_example['seg'], out_type = tf.float32)
    tes     = tf.io.parse_tensor(parsed_example['tes'], out_type = tf.float32)

    img_gt  = tf.reshape(img_gt, shape=[Nx,Ny,Nc])
    map_gt  = tf.reshape(map_gt, shape=[Nx,Ny,Np])
    seg     = tf.reshape(seg, shape=[Nx,Ny,1])
    tes     = tf.reshape(tes, shape=[Nc])

    return (img_gt,map_gt,seg,tes)

def parse_all(example):
    # parse training data.
    feature_discription = {
        'imgs_gt_bi':   tf.io.FixedLenFeature([], tf.string),
        'imgs_n_bi':    tf.io.FixedLenFeature([], tf.string),
        'imgs_gt_mono': tf.io.FixedLenFeature([], tf.string),
        'imgs_n_mono':  tf.io.FixedLenFeature([], tf.string),
        'maps_gt':      tf.io.FixedLenFeature([], tf.string),
        'seg':          tf.io.FixedLenFeature([], tf.string),
        'tes':          tf.io.FixedLenFeature([], tf.string),
        'Nx': tf.io.FixedLenFeature([], tf.int64),
        'Ny': tf.io.FixedLenFeature([], tf.int64),
        'Nc': tf.io.FixedLenFeature([], tf.int64),
        'Np': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed_example = tf.io.parse_single_example(example,feature_discription)

    Nx = parsed_example['Nx']
    Ny = parsed_example['Ny']
    Nc = parsed_example['Nc']
    Np = parsed_example['Np']

    imgs_gt_bi  = tf.io.parse_tensor(parsed_example['imgs_gt_bi'], out_type = tf.float32)
    imgs_n_bi   = tf.io.parse_tensor(parsed_example['imgs_n_bi'], out_type = tf.float32)
    imgs_gt_mono= tf.io.parse_tensor(parsed_example['imgs_gt_mono'], out_type = tf.float32)
    imgs_n_mono = tf.io.parse_tensor(parsed_example['imgs_n_mono'], out_type = tf.float32)
    maps_gt     = tf.io.parse_tensor(parsed_example['maps_gt'], out_type = tf.float32)
    seg         = tf.io.parse_tensor(parsed_example['seg'], out_type = tf.float32)
    tes         = tf.io.parse_tensor(parsed_example['tes'], out_type = tf.float32)

    imgs_gt_bi  = tf.reshape(imgs_gt_bi, shape=[Nx,Ny,Nc])
    imgs_n_bi   = tf.reshape(imgs_n_bi, shape=[Nx,Ny,Nc])
    imgs_gt_mono= tf.reshape(imgs_gt_mono, shape=[Nx,Ny,Nc])
    imgs_n_mono = tf.reshape(imgs_n_mono, shape=[Nx,Ny,Nc])
    maps_gt     = tf.reshape(maps_gt, shape=[Nx,Ny,Np])
    seg         = tf.reshape(seg, shape=[Nx,Ny,1])
    tes         = tf.reshape(tes, shape=[Nc])
    return (imgs_gt_bi,imgs_n_bi,imgs_gt_mono,imgs_n_mono,maps_gt,seg,tes)

def extract(imgs_gt_bi,imgs_n_bi,imgs_gt_mono,imgs_n_mono,maps_gt,seg,tau,
            id_tau=[],rescale=100.0,model_type='unrolling',data_type='mono',task='T2mapping'):
    '''
    Extract partial data to train the network, and rescale the parameters.
    - rescale, rescale the T2/T1 values to about 1, to accelerate the training.
    '''
    if len(id_tau) != 0: # if specific tau values was set
        imgs_gt_bi   = tf.gather(imgs_gt_bi,indices=id_tau,axis=-1)
        imgs_n_bi    = tf.gather(imgs_n_bi,indices=id_tau,axis=-1)
        imgs_gt_mono = tf.gather(imgs_gt_mono,indices=id_tau,axis=-1)
        imgs_n_mono  = tf.gather(imgs_n_mono,indices=id_tau,axis=-1)
        tau = tf.gather(tau,indices=id_tau,axis=-1)

    if task == 'T2mapping':
        A  = maps_gt[...,0]
        R2 = tf.math.divide_no_nan(rescale,maps_gt[...,1])
        maps_gt = tf.stack((A,R2),axis=-1)
    
    if task == 'T1mapping':
        A  = maps_gt[...,0]
        B  = maps_gt[...,1]
        R1 = tf.math.divide_no_nan(rescale,maps_gt[...,2])
        maps_gt = tf.stack((A,B,R1),axis=-1)
    tau = tau/rescale

    # maskout background
    mask = tf.where(seg>0.0,1.0,0.0)
    maps_gt = maps_gt*mask
    imgs_n_bi = imgs_n_bi*mask
    imgs_gt_bi = imgs_gt_bi*mask
    imgs_n_mono = imgs_n_mono*mask
    imgs_gt_mono = imgs_gt_mono*mask

    # mono-exponential or multi-exponential data.
    if data_type == 'multi': imgs_n,imgs_gt = imgs_n_bi,imgs_gt_bi
    if data_type == 'mono':  imgs_n,imgs_gt = imgs_n_mono,imgs_gt_mono
    # output data according to network type.
    if model_type == 'unrolling': return ((imgs_n,tau),maps_gt)
    if model_type == 'cnn':       return (imgs_n,maps_gt)
    if model_type == 'test':      return ((imgs_n,tau),maps_gt,seg)

def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir,'model_*.h5'))  # get name list of all .hdf5 files
    #file_list = os.listdir(save_dir)
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).h5.*",file_)
            epochs_exist.append(int(result[0]))
        initial_epoch=max(epochs_exist)   
    else:
        initial_epoch = 0
    return initial_epoch

def lr_schedule(epoch):
    initial_lr = 0.001
    if epoch<=10:
        lr = initial_lr
    elif epoch<=20:
        lr = initial_lr/2
    elif epoch<=40:
        lr = initial_lr/4
    elif epoch<=80:
        lr = initial_lr/8
    elif epoch<=160:
        lr = initial_lr/16
    else:
        lr = initial_lr/32 
    return lr

def get_len(dataset):
    """
    Get the length of TFdataset.
    """
    return sum(1 for _ in dataset)

class MSLE(tf.keras.losses.Loss):
    """
    Mean square error.
    """
    def call(self,y_true,y_pred):
        loss = tf.math.reduce_mean(tf.math.square(tf.math.subtract(tf.math.log(y_pred+1.0),tf.math.log(y_true+1.0))))
        return loss

class MSLE_mask(tf.keras.losses.Loss):
    """
    Mean square error.
    """
    def call(self,y_true,y_pred):
        mask    = tf.where(y_true>0.0, 1.0, 0.0)
        y_pred  = tf.math.multiply(y_pred,mask)
        y_true  = tf.math.multiply(y_true,mask)
        loss = tf.math.reduce_mean(tf.math.square(tf.math.subtract(tf.math.log(y_pred+1.0),tf.math.log(y_true+1.0))))
        return loss

class MSE(tf.keras.losses.Loss):
    """
    Mean square error.
    """
    def call(self,y_true,y_pred):
        loss = tf.math.reduce_mean(tf.math.square(tf.math.subtract(y_pred,y_true)))
        return loss

class MSE_mow(tf.keras.losses.Loss):
    """
    Mean Absolute Error.
    """
    def call(self,y_true,y_pred):
        Ns  = tf.cast(tf.shape(y_pred)[0],tf.float32)
        w   = tf.math.divide((tf.range(Ns,dtype=tf.float32)+1.0),Ns)
        e    = tf.math.square(tf.math.subtract(y_pred,y_true))
        loss = tf.math.reduce_mean(e,axis=[1,2,3,4])
        loss = tf.math.reduce_mean(loss*w)
        return loss

class MSE_mask(tf.keras.losses.Loss):
    """
    Masked Mean Square Error.
    """
    def call(self,y_true,y_pred):
        e    = tf.math.square(tf.math.subtract(y_pred,y_true))
        mask = tf.where(y_true>0.0, 1.0, 0.0)
        loss = tf.math.reduce_mean(tf.math.multiply(e,mask))
        return loss

class l2norm(tf.keras.losses.Loss):
    """
    L2 norm loss.
    """
    def call(self,y_true,y_pred):
        loss = tf.math.reduce_sum(tf.math.square(tf.math.subtract(y_pred,y_true)),axis=[-1,-2,-3])
        loss = tf.math.reduce_mean(loss)
        return loss

class l2norm_mow(tf.keras.losses.Loss):
    """
    Empirical weighted l2 norm loss over multiple output.
    """
    def call(self,y_true,y_pred):
        Ns      = tf.cast(tf.shape(y_pred)[0],tf.float32)
        w       = tf.math.divide((tf.range(Ns,dtype=tf.float32)+1.0),Ns)
        loss    = tf.math.reduce_sum(tf.math.square(tf.math.subtract(y_pred,y_true)),axis=[2,3,4])
        loss    = tf.math.reduce_mean(loss,axis=1)
        loss    = tf.math.reduce_mean(loss*w)
        return loss



class NRMSE(tf.keras.losses.Loss):
    """
    Averaged normalized root mean square error.
    """
    def call(self,y_true,y_pred):
        e  = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(tf.math.subtract(y_pred,y_true)),axis=[-2,-3]))
        gt = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(y_true),axis=[-2,-3]))
        nrmse  = tf.math.reduce_mean(tf.math.divide_no_nan(e,gt))
        return nrmse

class NRMSE_mask(tf.keras.losses.Loss):
    """
    Normalized riit mean square error. 
    """
    def call(self,y_true,y_pred):
        mask    = tf.where(y_true>0.0, 1.0, 0.0)
        y_pred  = tf.math.multiply(y_pred,mask)
        y_true  = tf.math.multiply(y_true,mask)
        e       = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(tf.math.subtract(y_pred,y_true)),axis=[-2,-3]))
        gt      = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(y_true),axis=[-2,-3]))
        nrmse   = tf.math.reduce_mean(tf.math.divide_no_nan(e,gt))
        return nrmse

class MAPE(tf.keras.losses.Loss):
    """
    Mean absolute percentage error.
    """
    def call(self,y_true,y_pred):
        e   = tf.math.abs(tf.math.subtract(y_pred,y_true))
        loss= tf.math.divide_no_nan(e,y_true)
        loss= tf.math.reduce_mean(loss)
        return loss

class MAPE_mow(tf.keras.losses.Loss):
    """
    Mean absolute percentage error.
    """
    def call(self,y_true,y_pred):
        Ns  = tf.cast(tf.shape(y_pred)[0],tf.float32)
        w   = tf.math.divide((tf.range(Ns,dtype=tf.float32)+1.0),Ns)
        e   = tf.math.abs(tf.math.subtract(y_pred,y_true))
        loss= tf.math.divide_no_nan(e,y_true)
        loss= tf.math.reduce_mean(loss,axis=[1,2,3,4])
        loss= tf.math.reduce_mean(loss*w)
        return loss

class MAE(tf.keras.losses.Loss):
    """
    Mean Absolute Error.
    """
    def call(self,y_true,y_pred):
        e    = tf.math.abs(tf.math.subtract(y_pred,y_true))
        loss = tf.math.reduce_mean(e)
        return loss

class l1norm(tf.keras.losses.Loss):
    """
    l1 norm.
    """
    def call(self,y_true,y_pred):
        e    = tf.math.abs(tf.math.subtract(y_pred,y_true))
        loss = tf.math.reduce_sum(e,axis=[-1,-2,-3])
        loss = tf.math.reduce_mean(loss)
        return loss

class l1norm_mow(tf.keras.losses.Loss):
    """
    l1 norm.
    """
    def call(self,y_true,y_pred):
        Ns  = tf.cast(tf.shape(y_pred)[0],tf.float32)
        w   = tf.math.divide((tf.range(Ns,dtype=tf.float32)+1.0),Ns)
        loss = tf.math.abs(tf.math.subtract(y_pred,y_true))
        loss = tf.math.reduce_sum(loss,axis=[2,3,4])
        loss = tf.math.reduce_mean(loss,axis=[1])
        loss = tf.math.reduce_mean(loss*w)
        return loss

class MAE_mow(tf.keras.losses.Loss):
    """
    Mean Absolute Error.
    """
    def call(self,y_true,y_pred):
        Ns  = tf.cast(tf.shape(y_pred)[0],tf.float32)
        w   = tf.math.divide((tf.range(Ns,dtype=tf.float32)+1.0),Ns)
        e    = tf.math.abs(tf.math.subtract(y_pred,y_true))
        loss = tf.math.reduce_mean(e,axis=[1,2,3,4])
        loss = tf.math.reduce_mean(loss*w)
        return loss

class MAE_mow_rev(tf.keras.losses.Loss):
    """
    Mean Absolute Error.
    """
    def call(self,y_true,y_pred):
        Ns = tf.cast(tf.shape(y_pred)[0],tf.float32)
        w  = tf.math.divide(tf.range(Ns,0,-1,dtype=tf.float32),Ns)
        e  = tf.math.abs(tf.math.subtract(y_pred,y_true))
        loss = tf.math.reduce_mean(e,axis=[1,2,3,4])
        loss = tf.math.reduce_mean(loss*w)
        return loss

class MAE_mask(tf.keras.losses.Loss):
    """
    Masked Mean Square Error.
    """
    def call(self,y_true,y_pred):
        e    = tf.math.abs(tf.math.subtract(y_pred,y_true))
        mask = tf.where(y_true>0.0, 1.0, 0.0)
        loss = tf.math.reduce_mean(tf.math.multiply(e,mask))
        return loss

def NRMSE_metric(y_true,y_pred):
    y_pred = y_pred[-1]
    mask    = tf.where(y_true>0.0, 1.0, 0.0)
    y_pred  = tf.math.multiply(y_pred,mask)
    y_true  = tf.math.multiply(y_true,mask)
    e  = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(tf.math.subtract(y_pred,y_true)),axis=[1,2]))
    gt = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(y_true),axis=[1,2]))
    nrmse  = tf.math.reduce_mean(tf.math.divide_no_nan(e,gt))
    return nrmse

def NRMSE_metric_cnn(y_true,y_pred):
    mask    = tf.where(y_true>0.0, 1.0, 0.0)
    y_pred  = tf.math.multiply(y_pred,mask)
    y_true  = tf.math.multiply(y_true,mask)
    e  = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(tf.math.subtract(y_pred,y_true)),axis=[1,2]))
    gt = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(y_true),axis=[1,2]))
    nrmse  = tf.math.reduce_mean(tf.math.divide_no_nan(e,gt))
    return nrmse

def display_image(imgs,filename=None):
    assert len(imgs.shape)==4, 'The input images should have 4 dims.'
    _,_,Np,Nc = imgs.shape
    row = int(np.ceil(np.sqrt(Np*Nc)))
    plt.figure(figsize=(row,row))
    i = 1
    for n in range(Np):
        for nc in range(Nc):
            plt.subplot(row,row,i),plt.imshow(imgs[:,:,n,nc],cmap='gray'),plt.axis('off')
            i=i+1
    plt.savefig(filename)

def display_func(func,filename=None):
    assert len(func.shape)==2, 'The function matrix should have 2 dims.'
    N,Nc = func.shape
    row = int(np.ceil(np.sqrt(N)))
    plt.figure(figsize=(row+1.0,row+1.0))
    i = 1
    for n in range(N):
        plt.subplot(row,row,i),plt.plot(func[n])
        plt.ylim((-1,1))
        i=i+1
    plt.savefig(filename)

class DisplayCallback(tf.keras.callbacks.Callback):
    def __init__(self,kernel_name=None,q_table_name=None):
        super(DisplayCallback,self).__init__()
        self.kernel_name  = kernel_name
        self.q_table_name = q_table_name
    def on_epoch_end(self,epoch,logs=None):
        kernel = [v for v in self.model.weights if v.name == self.kernel_name][0]
        func   = [v for v in self.model.weights if v.name == self.q_table_name][0]
        display_image(kernel,filename = os.path.join('figures','weights.png'))
        display_func(func,filename = os.path.join('figures','func.png'))

def nRMSE(y_pred,y_true,roi=None,imgae_axis=(1,2,3),average=True,deci=4):
    if roi is None: roi = np.where(y_true>0.0,1.0,0.0)
    y_pred = y_pred*roi
    y_true = y_true*roi
    e     = np.sqrt(np.sum(np.square(y_pred-y_true),axis=imgae_axis))
    nrmse = np.divide(e,np.sqrt(np.sum(np.square(y_true),axis=imgae_axis)))
    if average == True: nrmse = np.mean(nrmse)
    nrmse = np.round(nrmse,decimals=deci)
    return nrmse

def SSIM(y_pred,y_true,roi=None,data_range=None,average=True,deci=4):
    if roi is None: roi = np.where(y_true>0.0,1.0,0.0)
    y_pred = y_pred*roi
    y_true = y_true*roi
    ssim_maps = np.zeros_like(y_true)
    N,_,_,Nc  = y_pred.shape
    for i in range(N):
        for j in range(Nc):
            if data_range is None: data_range = np.max(y_true[i,:,:,j])-np.min(y_true[i,:,:,j])
            _,ssim_maps[i,:,:,j] = metric.structural_similarity(y_true[i,:,:,j],y_pred[i,:,:,j],\
                data_range=data_range,full=True,gaussian_weights=True,sigma=1.5,use_sample_covariance=False)
            # _,ssim_maps[i,:,:,j] = metric.structural_similarity(y_true[i,:,:,j],y_pred[i,:,:,j],data_range=data_range,full=True)
    ssim = np.divide(np.sum(ssim_maps*roi,axis=(1,2,3)),np.sum(roi,axis=(1,2,3)))
    if average==True: ssim = np.mean(ssim)
    ssim = np.round(ssim,decimals=deci)
    return ssim

def list_record_features(tfrecords_path):
    # Dict of extracted feature information
    features = {}
    # Iterate records
    for rec in tf.data.TFRecordDataset([str(tfrecords_path)]):
        # Get record bytes
        example_bytes = rec.numpy()
        # Parse example protobuf message
        example = tf.train.Example()
        example.ParseFromString(example_bytes)
        # Iterate example features
        for key, value in example.features.feature.items():
            # Kind of data in the feature
            kind = value.WhichOneof('kind')
            # Size of data in the feature
            size = len(getattr(value, kind).value)
            # Check if feature was seen before
            if key in features:
                # Check if values match, use None otherwise
                kind2, size2 = features[key]
                if kind != kind2:
                    kind = None
                if size != size2:
                    size = None
            # Save feature data
            features[key] = (kind, size)
    return features
