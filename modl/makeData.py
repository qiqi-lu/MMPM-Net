import os
import helper
import numpy as np
import matplotlib.pyplot as plt

# DATA PARAMETERS
print('='*100)
data_type ='LIVER'
# data_type ='PHANTOM'
pattern=1
# pattern=2
# pattern=3

noise_type='Gaussian'
# noise_type='Rician'
sigma=10

tes =np.array([0.93, 2.27, 3.61, 4.95, 6.29, 7.63, 8.97, 10.4, 11.8, 13.2, 14.6, 16.0]) # (London)
# tes =np.array([0.99, 2.40, 3.81, 5.22, 6.63, 8.04, 9.45, 10.86,12.27,13.68,15.09,16.5]) # (Chan2014)2
# tes =np.array([0.80, 1.05, 1.30, 1.55, 1.80, 2.05, 2.30, 2.55, 2.80, 3.05, 3.30, 3.55]) # (Wood2005)3


# SAVE TO
saveTo    = os.path.join('data',data_type+str(pattern))
saveToTrn = os.path.join(saveTo,noise_type,str(sigma),'TRAIN')
saveToTst = os.path.join(saveTo,noise_type,str(sigma),'TEST')

if not os.path.exists(saveToTrn):
    os.makedirs(saveToTrn)
if not os.path.exists(saveToTst):
    os.makedirs(saveToTst)

# PARAMETER MAPS
print('='*100)
print('Load (or create) parameter maps...')
if data_type=='LIVER':
    data = np.load(os.path.join('..','data_simulated','simulated_data_train_'+str(11)+'.npy'),allow_pickle=True).item()
    print('KEYS: ',data.keys())
    pImg = np.stack([data['s0 map'],data['r2s map']],-1)
elif data_type=='PHANTOM':
    size=(8,16)      # num of block
    size_block=(8,8) # block size
    n=121            # number of study
    pImg=np.zeros((n,size[0]*size_block[0],size[1]*size_block[1],2))
    for i in range(n):
        pImg[i,...,0],_ = helper.makeBlockImage(img_size=size,block_size=size_block,type='Random',value=[300,400]) # S0
        pImg[i,...,1],_ = helper.makeBlockImage(img_size=size,block_size=size_block,type='Random',value=[10,1000]) # R2
else:
    print('!! Unspported data type.')
print('Parameter map shape: ',pImg.shape)
np.save(os.path.join(saveTo,'pImg.npy'),pImg)

# show parameter maps
plt.figure(figsize=(30,20))
index = [2,25,50,120]
plt.subplot(2,4,1),plt.imshow(pImg[index[0],:,:,0],cmap='jet',vmax=450,vmin=0),plt.colorbar(fraction=0.024),plt.title('$S_0$')
plt.subplot(2,4,2),plt.imshow(pImg[index[0],:,:,1],cmap='jet',vmax=1050,vmin=0),plt.colorbar(fraction=0.024),plt.title('$R_2$')
plt.subplot(2,4,3),plt.imshow(pImg[index[1],:,:,0],cmap='jet',vmax=450,vmin=0),plt.colorbar(fraction=0.024)
plt.subplot(2,4,4),plt.imshow(pImg[index[1],:,:,1],cmap='jet',vmax=1050,vmin=0),plt.colorbar(fraction=0.024)
plt.subplot(2,4,5),plt.imshow(pImg[index[2],:,:,0],cmap='jet',vmax=450,vmin=0),plt.colorbar(fraction=0.024)
plt.subplot(2,4,6),plt.imshow(pImg[index[2],:,:,1],cmap='jet',vmax=1050,vmin=0),plt.colorbar(fraction=0.024)
plt.subplot(2,4,7),plt.imshow(pImg[index[3],:,:,0],cmap='jet',vmax=450,vmin=0),plt.colorbar(fraction=0.024)
plt.subplot(2,4,8),plt.imshow(pImg[index[3],:,:,1],cmap='jet',vmax=1050,vmin=0),plt.colorbar(fraction=0.024)
plt.savefig(os.path.join('figures','maps.png'))

# WEIGHTED IMAGES WITH OR WITHOUT NOISE
print('='*100)
wImg,wImgN = helper.makePairedData(pImg,tes=tes,sigma=sigma,noise_type=noise_type)
print('Weighted images shape:',wImgN.shape)

# TRAINING DATA
print('='*100)
print('Make training data...')
pImgTrn  = pImg[0:100]
wImgTrn  = wImg[0:100]
wImgNTrn = wImgN[0:100]

np.save(os.path.join(saveToTrn,'pImg.npy'),pImgTrn)
np.save(os.path.join(saveToTrn,'wImg.npy'),wImgTrn)
np.save(os.path.join(saveToTrn,'wImgN.npy'),wImgNTrn)

pPatch  = helper.makePatch(pImgTrn,rescale=False)
wPatch  = helper.makePatch(wImgTrn,rescale=False)
wPatchN = helper.makePatch(wImgNTrn,rescale=False)

np.save(os.path.join(saveToTrn,'pPatch.npy'),pPatch)
np.save(os.path.join(saveToTrn,'wPatch.npy'),wPatch)
np.save(os.path.join(saveToTrn,'wPatchN.npy'),wPatchN)

# TESTING DATA
print('='*100)
print('Make testing data...')
pImgTst  = pImg[100:]
wImgTst  = wImg[100:]
wImgNTst = wImgN[100:]

np.save(os.path.join(saveToTst,'pImg.npy'),pImgTst)
np.save(os.path.join(saveToTst,'wImg.npy'),wImgTst)
np.save(os.path.join(saveToTst,'wImgN.npy'),wImgNTst)

# MASK
print('='*100)
print('Make mask...')
if data_type=='LIVER':
    mask_body = np.load(os.path.join('..','data_clinical','clinical_data_bkg_mask_manual.npy'))
    mask_whole_liver = np.load(os.path.join('..','data_clinical','mask_liver_whole_manual.npy'))
    mask_parenchyma  = np.load(os.path.join('..','data_clinical','mask_liver_parenchyma_manual.npy'))
    mask_body=np.stack([mask_body,mask_body],axis=-1)
    mask_whole_liver=np.stack([mask_whole_liver,mask_whole_liver],axis=-1)
    mask_parenchyma=np.stack([mask_parenchyma,mask_parenchyma],axis=-1)
    maskTrn=mask_body[0:100]
    maskPatchTrn = helper.makePatch(maskTrn,rescale=False)
elif data_type=='PHANTOM':
    mask_body = np.ones(pImg.shape)
    maskPatchTrn = np.ones(pPatch.shape)

print('Mask shape: ',mask_body.shape)
print('Mask patch shape: ',maskPatchTrn.shape)

np.save(os.path.join(saveTo,'maskBody.npy'),mask_body)
np.save(os.path.join(saveTo,'maskLiver.npy'),mask_whole_liver)
np.save(os.path.join(saveTo,'maskParenchyma.npy'),mask_parenchyma)
np.save(os.path.join(saveTo,'maskPatchTrn.npy'),maskPatchTrn)


# show data
plt.figure(figsize=(20,15))
index = [52,2000]
m,n=6,4
plt.subplot(m,n,1),plt.imshow(wPatch[index[0],:,:,0],cmap='gray',vmax=450,vmin=0),plt.title('TE0'),plt.colorbar(fraction=0.023)
plt.subplot(m,n,2),plt.imshow(wPatch[index[0],:,:,1],cmap='gray',vmax=450,vmin=0),plt.title('TE1'),plt.colorbar(fraction=0.023)
plt.subplot(m,n,3),plt.imshow(wPatch[index[0],:,:,2],cmap='gray',vmax=450,vmin=0),plt.title('TE2'),plt.colorbar(fraction=0.023)
plt.subplot(m,n,4),plt.imshow(wPatch[index[0],:,:,3],cmap='gray',vmax=450,vmin=0),plt.title('TE3'),plt.colorbar(fraction=0.023)

plt.subplot(m,n,5),plt.imshow(wPatchN[index[0],:,:,0],cmap='gray',vmax=450,vmin=0),plt.colorbar(fraction=0.023)
plt.subplot(m,n,6),plt.imshow(wPatchN[index[0],:,:,1],cmap='gray',vmax=450,vmin=0),plt.colorbar(fraction=0.023)
plt.subplot(m,n,7),plt.imshow(wPatchN[index[0],:,:,2],cmap='gray',vmax=450,vmin=0),plt.colorbar(fraction=0.023)
plt.subplot(m,n,8),plt.imshow(wPatchN[index[0],:,:,3],cmap='gray',vmax=450,vmin=0),plt.colorbar(fraction=0.023)

plt.subplot(m,n,9),plt.imshow(pPatch[index[0],:,:,0],cmap='jet',vmax=350,vmin=0),plt.colorbar(fraction=0.023),plt.title('$S_0$')
plt.subplot(m,n,10),plt.imshow(pPatch[index[0],:,:,1],cmap='jet'),plt.colorbar(fraction=0.023),plt.title('$R_2$')
plt.subplot(m,n,11),plt.imshow(maskPatchTrn[index[0],:,:,0],cmap='jet'),plt.colorbar(fraction=0.023),plt.title('Mask')
# plt.subplot(m,n,12),plt.imshow(maskPatchTrn[index[0],:,:,1],cmap='jet'),plt.colorbar(fraction=0.023),plt.title('Mask')

plt.subplot(m,n,13),plt.imshow(wPatch[index[1],:,:,0],cmap='gray',vmax=450,vmin=0),plt.colorbar(fraction=0.023)
plt.subplot(m,n,14),plt.imshow(wPatch[index[1],:,:,1],cmap='gray',vmax=450,vmin=0),plt.colorbar(fraction=0.023)
plt.subplot(m,n,15),plt.imshow(wPatch[index[1],:,:,2],cmap='gray',vmax=450,vmin=0),plt.colorbar(fraction=0.023)
plt.subplot(m,n,16),plt.imshow(wPatch[index[1],:,:,3],cmap='gray',vmax=450,vmin=0),plt.colorbar(fraction=0.023)

plt.subplot(m,n,17),plt.imshow(wPatchN[index[1],:,:,0],cmap='gray',vmax=450,vmin=0),plt.colorbar(fraction=0.023)
plt.subplot(m,n,18),plt.imshow(wPatchN[index[1],:,:,1],cmap='gray',vmax=450,vmin=0),plt.colorbar(fraction=0.023)
plt.subplot(m,n,19),plt.imshow(wPatchN[index[1],:,:,2],cmap='gray',vmax=450,vmin=0),plt.colorbar(fraction=0.023)
plt.subplot(m,n,20),plt.imshow(wPatchN[index[1],:,:,3],cmap='gray',vmax=450,vmin=0),plt.colorbar(fraction=0.023)

plt.subplot(m,n,21),plt.imshow(pPatch[index[1],:,:,0],cmap='jet',vmax=350,vmin=0),plt.colorbar(fraction=0.023)
plt.subplot(m,n,22),plt.imshow(pPatch[index[1],:,:,1],cmap='jet'),plt.colorbar(fraction=0.023)
plt.subplot(m,n,23),plt.imshow(maskPatchTrn[index[1],:,:,0],cmap='jet'),plt.colorbar(fraction=0.023)
# plt.subplot(m,n,24),plt.imshow(maskPatchTrn[index[1],:,:,1],cmap='jet'),plt.colorbar(fraction=0.023)

plt.savefig(os.path.join('figures','patch.png'))

plt.figure()
index = [16]
m,n=6,4
plt.subplot(m,n,1),plt.imshow(wImg[index[0],:,:,0],cmap='gray',vmax=450,vmin=0),plt.title('TE0'),plt.colorbar(fraction=0.023)
plt.subplot(m,n,2),plt.imshow(wImg[index[0],:,:,1],cmap='gray',vmax=450,vmin=0),plt.title('TE1'),plt.colorbar(fraction=0.023)
plt.subplot(m,n,3),plt.imshow(wImg[index[0],:,:,2],cmap='gray',vmax=450,vmin=0),plt.title('TE2'),plt.colorbar(fraction=0.023)
plt.subplot(m,n,4),plt.imshow(wImg[index[0],:,:,3],cmap='gray',vmax=450,vmin=0),plt.title('TE3'),plt.colorbar(fraction=0.023)

plt.subplot(m,n,5),plt.imshow(wImgN[index[0],:,:,0],cmap='gray',vmax=450,vmin=0),plt.colorbar(fraction=0.023)
plt.subplot(m,n,6),plt.imshow(wImgN[index[0],:,:,1],cmap='gray',vmax=450,vmin=0),plt.colorbar(fraction=0.023)
plt.subplot(m,n,7),plt.imshow(wImgN[index[0],:,:,2],cmap='gray',vmax=450,vmin=0),plt.colorbar(fraction=0.023)
plt.subplot(m,n,8),plt.imshow(wImgN[index[0],:,:,3],cmap='gray',vmax=450,vmin=0),plt.colorbar(fraction=0.023)

plt.subplot(m,n,9),plt.imshow(pImg[index[0],:,:,0],cmap='jet',vmax=350,vmin=0),plt.colorbar(fraction=0.023),plt.title('$S_0$')
plt.subplot(m,n,10),plt.imshow(pImg[index[0],:,:,1],cmap='jet'),plt.colorbar(fraction=0.023),plt.title('$R_2$')
plt.subplot(m,n,11),plt.imshow(mask_body[index[0],:,:,0],cmap='jet'),plt.colorbar(fraction=0.023),plt.title('Mask')
plt.subplot(m,n,12),plt.imshow(mask_whole_liver[index[0],:,:,0],cmap='jet'),plt.colorbar(fraction=0.023),plt.title('Mask')
plt.savefig(os.path.join('figures','img.png'))
