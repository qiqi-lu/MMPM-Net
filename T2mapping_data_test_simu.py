import numpy as np
import matplotlib.pyplot as plt
import os
import functions as func
import config
import data_processor as dp

config.config_gpu(5)

#########################################################################################
subs_test = ['subject54']
region    = [165,215] # in [120,260]

TES       = {0 : np.arange(5.0,325.0,5.0)}
tes_keys  = [0]

# ---------------------------------------------------------------------------------------
dir_data  = os.path.join('data','testingdata')
dir_model = os.path.join('data','brainweb','anatomical_model') # BrainWeb database
subs      = os.listdir(dir_model) # all subjects in database
sigma     = 0
num_slice = region[-1]-region[0]

#########################################################################################
print('='*98)
for sub in subs_test:
    print('Subject: '+sub)
    if sub not in subs: 
        print(sub+' does not exist in database!')
        continue

    crisp_model_test, fuzzy_models_test = dp.read_models(dir_model=dir_model,sub_name=sub,ac=8) # real models
    fuzzy_models_test = fuzzy_models_test[region[0]:region[1]]
    crisp_model_test  = crisp_model_test[region[0]:region[1]]

    fuzzy_models_test_masked = dp.maskBKG(fuzzy_models_test,crisp_model_test)
    crisp_model_test         = crisp_model_test[...,np.newaxis].astype(np.float32)

# ---------------------------------------------------------------------------------------
    for key in tes_keys:
        tes_test     = TES[key] # (ms)
        print('tes: '+str(tes_test))

        para_type    = 'time'
        imgs_gt_test = dp.model2image(sub_name=sub, model=fuzzy_models_test_masked,tau=tes_test,fluctuation=True,fraction=0.025,ac=4)
        maps_gt_test = dp.image2map(imgs=imgs_gt_test,tau=tes_test,parameter_type=para_type,algorithm='NLLS',ac=1) # fitted map

        imgs_gt_test = imgs_gt_test.astype(np.float32)
        maps_gt_test = maps_gt_test.astype(np.float32)
        tes_test     = np.repeat(np.reshape(tes_test,(1,-1)),repeats=imgs_gt_test.shape[0],axis=0).astype(np.float32)

        # save as tfrecord data
        func.write2TFRecord_noise_free(img_gt=imgs_gt_test,map_gt=maps_gt_test,seg=crisp_model_test,tes=tes_test,\
                            filename=os.path.join(dir_data,'{}_te_{}_sigma_{}_S_{}'.format(sub,key,sigma,num_slice)))
    print('-'*98)

#########################################################################################
print('='*98)
fig,axes = plt.subplots(nrows=3,ncols=5,figsize=(15,9),dpi=300)
idx = 0
[ax.set_axis_off() for ax in axes.ravel()]
for i in range(5):
    axes[0,i].imshow(fuzzy_models_test_masked[idx,...,i],cmap='hot',vmin=0.0,vmax=1.0)
    axes[1,i].imshow(imgs_gt_test[idx,...,i],cmap='gray',vmin=0.0,vmax=1.2)
    axes[1,i].set_title('Noise free (TE'+str(i)+')')

axes[2,0].imshow(maps_gt_test[idx,...,0],cmap='gray',vmin=0.0,vmax=1.2)
axes[2,1].imshow(maps_gt_test[idx,...,1],cmap='jet',vmin=0.0,vmax=300.0)
axes[2,2].imshow(1000.0/maps_gt_test[idx,...,1],cmap='jet',vmin=0.0,vmax=20.0)
axes[2,3].imshow(crisp_model_test[idx],cmap='jet')
plt.savefig('figures/data_test')

#########################################################################################
print('='*98)

