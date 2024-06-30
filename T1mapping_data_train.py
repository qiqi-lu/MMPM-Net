# R1 mapping task.
# Training data simulation.

import numpy as np
import matplotlib.pyplot as plt
import os
import data_processor as dp
import functions as func

##########################################################################################
subs_train = ['subject45']
tau = np.array([50.0,100.0,200.0,400.0,800.0,1600.0,3200.0,6400.0]) # Invertion Time (TI), ms
id_tau  = 1
region = [120,260]
patch_size = 64
step_size = 32

# -----------------------------------------------------------------------------------------
dir_data  = os.path.join('data','T1mapping','trainingdata') # save to
dir_model = os.path.join('data','brainweb','anatomical_model') # BrainWeb database
subs      = os.listdir(dir_model) # all subjects in database
dir_fig   = os.path.join('figures','T1mapping') # figure folder
sigma     = 0 # noise-free datasets
num_slice = region[-1]-region[0]

##########################################################################################
print('='*98)
for sub in subs_train:
    print('-'*98)
    print('Subject: '+sub)
    # check subject existence
    if sub not in subs: 
        print(sub+' does not exist in database!')
        continue
    # read modelss
    crisp_model_train, fuzzy_models_train = dp.read_models(dir_model=dir_model,sub_name=sub,ac=8) 
    fuzzy_models_train  = fuzzy_models_train[region[0]:region[1]]
    crisp_model_train   = crisp_model_train [region[0]:region[1]]
    # mask out background 
    fuzzy_models_train  = dp.maskBKG(fuzzy_models_train,crisp_model_train)
    # make model patches
    fuzzy_model_patch   = func.patch(data=fuzzy_models_train,mask=crisp_model_train,patch_size=patch_size,step_size=step_size)
    crisp_model_patch   = func.patch(data=crisp_model_train[...,np.newaxis],mask=crisp_model_train,patch_size=patch_size,step_size=step_size)
    crisp_model_patch   = crisp_model_patch.astype(np.float32)
    # convert model to images and maps
    print('TI (ms): '+str(tau))
    para_type     = 'time'
    imgs_gt_train = dp.model_to_T1w_image(sub_name=sub,model=fuzzy_model_patch,tau=tau,fraction=0.025,ac=8) # model -> image
    maps_gt_train = dp.image2map(imgs=imgs_gt_train,tau=tau,fitting_model='T1_three_para_magn',signed=True,parameter_type=para_type,
                                 algorithm='NLLS',pbar_disable=True,ac=8) # image -> map
    # save data into TFRecord file
    imgs_gt_train = imgs_gt_train.astype(np.float32)
    maps_gt_train = maps_gt_train.astype(np.float32)
    tau_train     = np.repeat(np.reshape(tau,(1,-1)),repeats=imgs_gt_train.shape[0],axis=0).astype(np.float32)
    func.write2TFRecord_noise_free(img_gt=imgs_gt_train,map_gt=maps_gt_train,seg=crisp_model_patch,tes=tau_train,\
                                    filename=os.path.join(dir_data,'{}_te_{}_sigma_{}_S_{}'.format(sub,id_tau,sigma,num_slice)))

##############################################################################
#### show discrete and fuzzy models
print('='*98)
Nm = fuzzy_models_train.shape[-1]
id_show = 0
colume_width = 3.5 # inches
page_width = 7.16 # inches

# discrete model
fig,axes=plt.subplots(nrows=1,ncols=1,dpi=600,tight_layout=True,figsize=(colume_width,colume_width))
axes.set_axis_off()
axes.set_title('discrete model')
axes.imshow(crisp_model_train[id_show,:,:],cmap='hot')
plt.savefig(os.path.join(dir_fig,'model_discrete.png'))

# fuzzy model
fig,axes=plt.subplots(nrows=3,ncols=4,dpi=600,tight_layout=True,figsize=(page_width,page_width/4*3))
axes = axes.ravel()
for i in range(Nm):
    axes[i].set_axis_off()
    axes[i].imshow(fuzzy_models_train[id_show,:,:,i],vmin=0.0,vmax=1.0,cmap='hot')
plt.savefig(os.path.join(dir_fig,'model_fuzzy.png'))

# generated patches.
fig,axes=plt.subplots(nrows=4,ncols=12,dpi=600,tight_layout=True,figsize=(page_width,page_width/3))
for i in range(Nm):
    for j in range(4):
        axes[j,i].set_axis_off()
        axes[j,i].imshow(fuzzy_model_patch[j,:,:,i],vmin=0.0,vmax=1.0,cmap='hot')
plt.savefig(os.path.join(dir_fig,'model_patch.png'))

##############################################################################

    
