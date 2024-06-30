import numpy as np
import matplotlib.pyplot as plt
import os
import functions as func
import data_processor as dp
import config

config.config_gpu(5)
# np.random.seed(seed=8200)

##########################################################################################
subs_train = ['subject52','subject53']
tes  = np.arange(10.0,330.0,10.0)
id_te      = 1
region     = [120,260] # slice region unsampeled
patch_size = 64
step_size  = 32

# -----------------------------------------------------------------------------------------
dir_data  = os.path.join('data','trainingdata')
dir_model = os.path.join('data','brainweb','anatomical_model') # BrainWeb database
subs      = os.listdir(dir_model) # all subjects in database
dir_fig   = 'figures'
sigma     = 0
num_slice = region[-1]-region[0]

##########################################################################################
print('='*98)
for sub in subs_train:
    print('-'*98)
    print('Subject: '+sub)
    if sub not in subs: 
        print(sub+' does not exist in database!')
        continue

    # Read models.
    crisp_model_train, fuzzy_models_train = dp.read_models(dir_model=dir_model,sub_name=sub,ac=8) 
    fuzzy_models_train  = fuzzy_models_train[region[0]:region[1]]
    crisp_model_train   = crisp_model_train [region[0]:region[1]]

    fuzzy_models_train  = dp.maskBKG(fuzzy_models_train,crisp_model_train) # Mask out background.

    # Patching models.
    print('-'*98)
    print('Patching ...')
    fuzzy_model_patch   = func.patch(data=fuzzy_models_train,mask=crisp_model_train,patch_size=patch_size,step_size=step_size)
    crisp_model_patch   = func.patch(data=crisp_model_train[...,np.newaxis],mask=crisp_model_train,patch_size=patch_size,step_size=step_size)
    crisp_model_patch   = crisp_model_patch.astype(np.float32)

    print('TE: '+str(tes))

    para_type     = 'time'
    imgs_gt_train = dp.model2image(sub_name=sub,model=fuzzy_model_patch,tau=tes,fluctuation=True,fraction=0.025,ac=8) # model -> image
    maps_gt_train = dp.image2map(imgs=imgs_gt_train,tau=tes,parameter_type=para_type,algorithm='NLLS',pbar_disable=True,ac=8)        # image -> map

    imgs_gt_train = imgs_gt_train.astype(np.float32)
    maps_gt_train = maps_gt_train.astype(np.float32)
    tes_train     = np.repeat(np.reshape(tes,(1,-1)),repeats=imgs_gt_train.shape[0],axis=0).astype(np.float32)
    func.write2TFRecord_noise_free(img_gt=imgs_gt_train,map_gt=maps_gt_train,seg=crisp_model_patch,tes=tes_train,\
                                    filename=os.path.join(dir_data,'{}_te_{}_sigma_{}_S_{}'.format(sub,id_te,sigma,num_slice)))

##############################################################################
#### show discrete and fuzzy models
print('='*98)
Nm = fuzzy_models_train.shape[-1]
id = 0
plt.figure(figsize=(5,5),dpi=300)
plt.imshow(crisp_model_train[id,:,:],cmap='hot'),plt.title('discrete model'),plt.axis('off'),plt.colorbar(fraction=0.022)
plt.savefig(os.path.join(dir_fig,'model_discrete.png'))

plt.figure(figsize=(15,15),dpi=300)
for i in range(Nm):
    plt.subplot(3,4,i+1),plt.imshow(fuzzy_models_train[id,:,:,i],vmin=0.0,vmax=1.0,cmap='hot'),plt.axis('off'),plt.colorbar(fraction=0.022)
plt.savefig(os.path.join(dir_fig,'model_fuzzy.png'))

plt.figure(figsize=(12,4),dpi=300)
for i in range(Nm):
    for j in range(4):
        plt.subplot(4,Nm,i+1+j*Nm),plt.imshow(fuzzy_model_patch[j,:,:,i],vmin=0.0,vmax=1.0,cmap='hot'),plt.axis('off')
plt.savefig(os.path.join(dir_fig,'model_patch.png'))

##############################################################################
