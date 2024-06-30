import os
os.environ["OMP_NUM_THREADS"]        = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"]   = "1" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"]        = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"]    = "1" # export NUMEXPR_NUM_THREADS=6

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import config
import functions as func
import tqdm
# import model_based_recon as more
import model_based_recon_t2 as more

config.config_gpu(2)

print('-'*98)
subject_idx = 51
sigma       = 0.05
te_pattern  = 1

dataset_test_name = tf.io.gfile.glob(os.path.join('data','testingdata','*'+str(subject_idx)+'*'+str(sigma)+'.tfrecordsbm'))
# dataset_test      = tf.data.TFRecordDataset(dataset_test_name).map(func._parse_function_test_t2)
dataset_test      = tf.data.TFRecordDataset(dataset_test_name).map(func._parse_function_test_t2_mono)
dataset_test_size = func.get_len(dataset_test)

start_idx = 65
take_size = 1

dataset_test = dataset_test.skip(start_idx).batch(take_size).take(1)
for sample_all in dataset_test:
    inputs  = sample_all[0]
    imgs    = inputs[0]
    tes     = inputs[1]
    maps_gt = sample_all[1]

TEs        = tes.numpy()[0].astype(np.float64)
imgs       = imgs.numpy().astype(np.float64)
maps_gt    = maps_gt.numpy().astype(np.float64)

print('Test datasets:')
print(dataset_test_name)
print('Num of total data: '+str(dataset_test_size))
print('imgs: '+str(imgs.shape))
print('tes : '+str(TEs.shape)+' '+str(TEs))
print('-'*98)

##### Reconstruction #####
N,Ny,Nx,Nc = imgs.shape

remake = True
# remake = False

#############################################################################
print('===== Pixel-wise Fitting (PWF) =====')
if remake == True:
    maps_pwf = []
    pbar     = tqdm.tqdm(desc='PWF',total=N)

    for i in range(N):
        pbar.update(1)
        map = func.PixelWiseMapping(imgs[i],tes=TEs,model='EXP',pbar_leave=False)
        maps_pwf.append(map)
    pbar.close()
    maps_pwf = np.array(maps_pwf)
    np.save(os.path.join('data','testingdata','subject'+str(subject_idx)+'_te_'+str(te_pattern)+'_sigma_'+str(sigma)+'_map_pwf'),maps_pwf)

if remake == False:
    maps_pwf = np.load(os.path.join('data','testingdata','subject'+str(subject_idx)+'_te_'+str(te_pattern)+'_sigma_'+str(sigma)+'_map_pwf.npy'))

print("===== Pixelwise curve fitting with adaptive neibourhood regularization (PCANR) =====")
if remake == True:
    maps_pcanr = []
    pbar     = tqdm.tqdm(desc='PCANR',total=N)

    for i in range(N):
        pbar.update(1)
        map = func.PCANR(imgs[i],tes=TEs,h=0.2,f=7,m=0,Ncoils=1,pbar_leave=False)
        maps_pcanr.append(map)
    pbar.close()
    maps_pcanr = np.array(maps_pcanr)
    np.save(os.path.join('data','testingdata','subject'+str(subject_idx)+'_te_'+str(te_pattern)+'_sigma_'+str(sigma)+'_map_pcanr'),maps_pcanr)

if remake == False:
    maps_pcanr = np.load(os.path.join('data','testingdata','subject'+str(subject_idx)+'_te_'+str(te_pattern)+'_sigma_'+str(sigma)+'_map_pcanr.npy'))

print('===== CS-MRI reconstruction =====')
if remake == True:
    maps_cs_dwt = []
    pbar     = tqdm.tqdm(desc='CS-DWT',total=N)

    for i in range(N):
        pbar.update(1)
        x0 = imgs[i,...,0:2]*0.0+1.0
        scale = 1.0
        map = more.cs_mri_parameter(b=imgs[i],tes=TEs,DC='L2',RE='DWT',lambda_1=0.05,x_init=x0,MaxIter=20000,scale=scale)
        maps_cs_dwt.append(map)
    pbar.close()
    maps_cs_dwt = np.array(maps_cs_dwt)
    np.save(os.path.join('data','testingdata','subject'+str(subject_idx)+'_te_'+str(te_pattern)+'_sigma_'+str(sigma)+'_map_cs_dwt'),maps_cs_dwt)

if remake == False:
    maps_cs_dwt = np.load(os.path.join('data','testingdata','subject'+str(subject_idx)+'_te_'+str(te_pattern)+'_sigma_'+str(sigma)+'_map_cs_dwt.npy'))
###################################################################
def rescale(maps,scale):
    maps[...,1] = maps[...,1]*scale
    return maps

scale = 100.0
maps_pwf    = rescale(maps_pwf,scale)
maps_pcanr  = rescale(maps_pcanr,scale)
maps_cs_dwt = rescale(maps_cs_dwt,scale)
maps_gt     = rescale(maps_gt,scale)

# Evaluation
nrmse_m0_pwf    = np.round(func.nRMSE(y_pred=maps_pwf[...,0][...,np.newaxis],y_true=maps_gt[...,0][...,np.newaxis],mask=None,average=False),decimals=4)
nrmse_m0_pcanr  = np.round(func.nRMSE(y_pred=maps_pcanr[...,0][...,np.newaxis],y_true=maps_gt[...,0][...,np.newaxis],mask=None,average=False),decimals=4)
nrmse_m0_cs_dwt = np.round(func.nRMSE(y_pred=maps_cs_dwt[...,0][...,np.newaxis],y_true=maps_gt[...,0][...,np.newaxis],mask=None,average=False),decimals=4)

nrmse_t2_pwf    = np.round(func.nRMSE(y_pred=maps_pwf[...,1][...,np.newaxis],y_true=maps_gt[...,1][...,np.newaxis],mask=None,average=False),decimals=4)
nrmse_t2_pcanr  = np.round(func.nRMSE(y_pred=maps_pcanr[...,1][...,np.newaxis],y_true=maps_gt[...,1][...,np.newaxis],mask=None,average=False),decimals=4)
nrmse_t2_cs_dwt = np.round(func.nRMSE(y_pred=maps_cs_dwt[...,1][...,np.newaxis],y_true=maps_gt[...,1][...,np.newaxis],mask=None,average=False),decimals=4)

ssim_m0_pwf    = np.round(func.SSIM( y_pred=maps_pwf[...,0][...,np.newaxis],y_true=maps_gt[...,0][...,np.newaxis],mask=None,average=False),decimals=4)
ssim_m0_pcanr  = np.round(func.SSIM( y_pred=maps_pcanr[...,0][...,np.newaxis],y_true=maps_gt[...,0][...,np.newaxis],mask=None,average=False),decimals=4)
ssim_m0_cs_dwt = np.round(func.SSIM( y_pred=maps_cs_dwt[...,0][...,np.newaxis],y_true=maps_gt[...,0][...,np.newaxis],mask=None,average=False),decimals=4)

ssim_t2_pwf    = np.round(func.SSIM( y_pred=maps_pwf[...,1][...,np.newaxis],y_true=maps_gt[...,1][...,np.newaxis],mask=None,average=False),decimals=4)
ssim_t2_pcanr  = np.round(func.SSIM( y_pred=maps_pcanr[...,1][...,np.newaxis],y_true=maps_gt[...,1][...,np.newaxis],mask=None,average=False),decimals=4)
ssim_t2_cs_dwt = np.round(func.SSIM( y_pred=maps_cs_dwt[...,1][...,np.newaxis],y_true=maps_gt[...,1][...,np.newaxis],mask=None,average=False),decimals=4)

# print('-'*98+'\nEvaluation:')
# print('maps: '+str(maps_pwf.shape))
# print('NRMSE(M0|T2): '+str(np.round(np.mean(nrmse_m0_pwf),4))+'('+str(np.round(np.std(nrmse_m0_pwf),4))+') | '
#                       +str(np.round(np.mean(nrmse_t2_pwf),4))+'('+str(np.round(np.std(nrmse_t2_pwf),4))+')')
# print('SSIM(M0|T2) : '+str(np.round(np.mean(ssim_m0_pwf ),4))+'('+str(np.round(np.std(ssim_m0_pwf ),4))+') | '
#                       +str(np.round(np.mean(ssim_t2_pwf ),4))+'('+str(np.round(np.std(ssim_t2_pwf ),4))+')')

mask = np.where(maps_gt>0.0,1.0,0.0)
maps_pwf    = maps_pwf*mask
maps_pcanr  = maps_pcanr*mask
maps_cs_dwt = maps_cs_dwt*mask

print('-'*98)
print('Show results...')
row = 5
col = 4
slice_idx = -1
plt.figure(figsize=(col*4,row*4),dpi=300)
# row 1
for i in range(int(min(Nc,col))):
    plt.subplot(row,col,i+1)
    plt.imshow(imgs[slice_idx,...,i],cmap='gray',vmin=0.0,vmax=1.3),plt.colorbar(fraction=0.022),plt.title('TE='+str(TEs[i])),plt.axis('off')

# row 2
plt.subplot(row,col,col*1+1)
plt.imshow(maps_gt[slice_idx,...,0],cmap='gray',vmin=0.0,vmax=1.3),plt.colorbar(fraction=0.022),plt.title('$S_0 (GT)$'),plt.axis('off')
plt.subplot(row,col,col*1+2)
plt.imshow(maps_pwf[slice_idx,...,0],cmap='gray',vmin=0.0,vmax=1.3),plt.colorbar(fraction=0.022),plt.title('$S_0 (PWF)$'),plt.axis('off')
plt.text(x=5,y=15,s=str(ssim_m0_pwf[slice_idx]),color='white',fontsize=15.0)
plt.subplot(row,col,col*1+3)
plt.imshow(maps_pcanr[slice_idx,...,0],cmap='gray',vmin=0.0,vmax=1.3),plt.colorbar(fraction=0.022),plt.title('$S_0 (PCANR)$'),plt.axis('off')
plt.text(x=5,y=15,s=str(ssim_m0_pcanr[slice_idx]),color='white',fontsize=15.0)
plt.subplot(row,col,col*1+4)
plt.imshow(maps_cs_dwt[slice_idx,...,0],cmap='gray',vmin=0.0,vmax=1.3),plt.colorbar(fraction=0.022),plt.title('$S_0 (CS-DWT)$'),plt.axis('off')
plt.text(x=5,y=15,s=str(ssim_m0_cs_dwt[slice_idx]),color='white',fontsize=15.0)

# row 3
plt.subplot(row,col,col*2+2)
plt.imshow(tf.math.divide_no_nan(tf.math.abs(maps_gt[slice_idx,...,0]-maps_pwf[slice_idx,...,0]),maps_gt[slice_idx,...,0]),cmap='jet',vmin=0.0,vmax=0.2),plt.colorbar(fraction=0.022),plt.title('RE'),plt.axis('off')
plt.text(x=5,y=15,s=str(nrmse_m0_pwf[slice_idx]),color='white',fontsize=15.0)
plt.subplot(row,col,col*2+3)
plt.imshow(tf.math.divide_no_nan(tf.math.abs(maps_gt[slice_idx,...,0]-maps_pcanr[slice_idx,...,0]),maps_gt[slice_idx,...,0]),cmap='jet',vmin=0.0,vmax=0.2),plt.colorbar(fraction=0.022),plt.title('RE'),plt.axis('off')
plt.text(x=5,y=15,s=str(nrmse_m0_pcanr[slice_idx]),color='white',fontsize=15.0)
plt.subplot(row,col,col*2+4)
plt.imshow(tf.math.divide_no_nan(tf.math.abs(maps_gt[slice_idx,...,0]-maps_cs_dwt[slice_idx,...,0]),maps_gt[slice_idx,...,0]),cmap='jet',vmin=0.0,vmax=0.2),plt.colorbar(fraction=0.022),plt.title('RE'),plt.axis('off')
plt.text(x=5,y=15,s=str(nrmse_m0_cs_dwt[slice_idx]),color='white',fontsize=15.0)

# row 4
plt.subplot(row,col,col*3+1)
plt.imshow(maps_gt[slice_idx,...,1],cmap='jet',vmin=0.0,vmax=300.0),plt.colorbar(fraction=0.022),plt.title('$T_2 (GT)$'),plt.axis('off')
plt.subplot(row,col,col*3+2)
plt.imshow(maps_pwf[slice_idx,...,1],cmap='jet',vmin=0.0,vmax=300.0),plt.colorbar(fraction=0.022),plt.title('$T_2 (PWF)$'),plt.axis('off')
plt.text(x=5,y=15,s=str(ssim_t2_pwf[slice_idx]),color='white',fontsize=15.0)
plt.subplot(row,col,col*3+3)
plt.imshow(maps_pcanr[slice_idx,...,1],cmap='jet',vmin=0.0,vmax=300.0),plt.colorbar(fraction=0.022),plt.title('$T_2 (PCANR)$'),plt.axis('off')
plt.text(x=5,y=15,s=str(ssim_t2_pcanr[slice_idx]),color='white',fontsize=15.0)
plt.subplot(row,col,col*3+4)
plt.imshow(maps_cs_dwt[slice_idx,...,1],cmap='jet',vmin=0.0,vmax=300.0),plt.colorbar(fraction=0.022),plt.title('$T_2 (CS-DWT)$'),plt.axis('off')
plt.text(x=5,y=15,s=str(ssim_t2_cs_dwt[slice_idx]),color='white',fontsize=15.0)

# row 5
plt.subplot(row,col,col*4+2)
plt.imshow(tf.math.divide_no_nan(tf.math.abs(maps_gt[slice_idx,...,1]-maps_pwf[slice_idx,...,1]),maps_gt[slice_idx,...,1]),cmap='jet',vmin=0.0,vmax=0.5),plt.colorbar(fraction=0.022),plt.title('RE'),plt.axis('off')
plt.text(x=5,y=15,s=str(nrmse_t2_pwf[slice_idx]),color='white',fontsize=15.0)
plt.subplot(row,col,col*4+3)
plt.imshow(tf.math.divide_no_nan(tf.math.abs(maps_gt[slice_idx,...,1]-maps_pcanr[slice_idx,...,1]),maps_gt[slice_idx,...,1]),cmap='jet',vmin=0.0,vmax=0.5),plt.colorbar(fraction=0.022),plt.title('RE'),plt.axis('off')
plt.text(x=5,y=15,s=str(nrmse_t2_pcanr[slice_idx]),color='white',fontsize=15.0)
plt.subplot(row,col,col*4+4)
plt.imshow(tf.math.divide_no_nan(tf.math.abs(maps_gt[slice_idx,...,1]-maps_cs_dwt[slice_idx,...,1]),maps_gt[slice_idx,...,1]),cmap='jet',vmin=0.0,vmax=0.5),plt.colorbar(fraction=0.022),plt.title('RE'),plt.axis('off')
plt.text(x=5,y=15,s=str(nrmse_t2_cs_dwt[slice_idx]),color='white',fontsize=15.0)

plt.tight_layout()
plt.savefig(os.path.join('figures','maps_model_based_recon'))

print('-'*98)