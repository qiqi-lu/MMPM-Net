# custom metrics
import numpy as np
from scipy import stats

def nRMSE(x,y,mask=None,mean=False):
    '''
    Normalized relative mean square error metric.
    ### AUGMENTS
    x: refrence data with a shape of [batch,w,h,channel] or [batch,w,h].
    y: predicted data.
    mask: mask applied on the data.
    ### RETURN
    nrmse: nrmse value(s).
    '''
    assert x.shape == y.shape, "The input data shapes are un-consistent."
    if len(x.shape) == 3:
        x=x[...,np.newaxis]
        y=y[...,np.newaxis]

    nrmse = []
    (n,w,h,c) = x.shape

    if mask is not None:
        if len(mask.shape)==3: mask = mask[...,np.newaxis]
        mask = np.repeat(mask,c,axis=-1)
        x=x*mask
        y=y*mask
    for i in range(n):
        e = np.linalg.norm(x[i]-y[i])/np.linalg.norm(x[i])
        nrmse.append(e)
    if mean == True: nrmse = np.mean(nrmse) # mean nrmse
    nrmse = np.round(nrmse,4)
    return nrmse

def SSIM(x,y,mask=None,mean=False,data_range=1024):
    """
    Structure similarity metric.
    ### AUGMENTS
    x: Refrence data with a shape of [batch, w, h, channel] or [batch,w,h].
    y: Predicted data.
    mask: Mask applied on the data.
    mean: Mean of all.
    data_range: The data range of the input image.
    ### RETURN
    nrmse: ssim value(s).
    """
    from skimage import metrics
    assert x.shape == y.shape, "The input data shapes are un-consistent."
    if len(x.shape) == 3:
        x=x[...,np.newaxis]
        y=y[...,np.newaxis]

    ssim = []
    (n,w,h,c) = x.shape

    if mask is not None:
        if len(mask.shape)==3: mask = mask[...,np.newaxis]
        mask = np.repeat(mask,c,axis=-1)
    
    for i in range(n):
        mssim,ssim_map = metrics.structural_similarity(x[i], y[i],data_range=data_range,full=True,multichannel=True)
        if mask is not None:
            ss = np.sum(ssim_map*mask[i])/np.sum(mask[i])
        ssim.append(mssim)
    if mean == True: ssim = np.mean(ssim)
    ssim = np.round(ssim,4)
    return ssim

def PSNR(x,y,mean=False,data_range=1):
    """
    Peak signal to noise ratio.
    ### AUGMENTS
    x: Refrence data with a shape of [batch, w, h, channel] or [batch,w,h].
    y: Predicted data.
    mean: Mean of all.
    data_range: The data range of the input image.
    ### RETURN
    psnr: PSNR value(s).
    """
    from skimage import metrics
    assert x.shape == y.shape, "The input data shapes are un-consistent."
    if len(x.shape) == 3:
        x=x[...,np.newaxis]
        y=y[...,np.newaxis]

    psnr = []
    (n,w,h,c) = x.shape

    for i in range(n):
        r = metrics.peak_signal_noise_ratio(x[i], y[i],data_range=data_range)
        psnr.append(r)
    if mean == True: psnr = np.mean(psnr)
    psnr = np.round(psnr,4)
    return psnr

def Pvalue(A,B,alt='less'):
    """
    P value between two vector.
    ### AUGMENTS
    A, B: vector.
    ### RETURN
    p, pvalue.
    """
    a = np.array(A)
    b = np.array(B)
    w,p = stats.wilcoxon(a,b,correction=True, alternative=alt)
    return p