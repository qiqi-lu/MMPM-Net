import numpy as np
import matplotlib.pyplot as plt
import os
import functions as func
# import config
import data_processor as dp

# config.config_gpu(5)

#########################################################################################
subs_test = ['subject54']
# region    = [165,215] # in [120,260], slices used for testing
# region    = [185,186]
region    = [185,186]
sigma     = 0

tau_test  = [50.0,  100.0,  150.0,  200.0,  250.0,  300.0,  350.0,  400.0,  450.0,  500.0,  550.0,  600.0,  650.0,  700.0,  750.0,  800.0,
             850.0, 900.0,  950.0,  1000.0, 1050.0, 1100.0, 1150.0, 1200.0, 1250.0, 1300.0, 1350.0, 1400.0, 1450.0, 1500.0, 1550.0, 1600.0,
             1650.0,1700.0, 1750.0, 1800.0, 1850.0, 1900.0, 1950.0, 2000.0, 2050.0, 2100.0, 2150.0, 2200.0, 2250.0, 2300.0, 2350.0, 2400.0,
             2450.0,2500.0, 2600.0, 2700.0, 2800.0, 2900.0, 3000.0, 3100.0, 3200.0, 3300.0, 3400.0, 3500.0, 3600.0, 3700.0, 3800.0, 3900.0,
             4000.0,4200.0, 4400.0, 4600.0, 4800.0, 5000.0, 5200.0, 5400.0, 5600.0, 5800.0, 6000.0, 6400.0, 7000.0, 8000.0, 9000.0, 10000.0]
tau_test = np.array(tau_test)
# ---------------------------------------------------------------------------------------
dir_data  = os.path.join('data','T1mapping','testingdata')
dir_model = os.path.join('data','brainweb','anatomical_model') # BrainWeb database
subs      = os.listdir(dir_model) # all subjects in database
num_slice = region[-1]-region[0]

#########################################################################################
print('='*98)
for sub in subs_test:
    print('Subject: '+sub)
    if sub not in subs: 
        print(sub+' does not exist in database!')
        continue
    # load models
    crisp_model_test, fuzzy_models_test = dp.read_models(dir_model=dir_model,sub_name=sub,ac=8)
    fuzzy_models_test = fuzzy_models_test[region[0]:region[1]]
    crisp_model_test  = crisp_model_test[region[0]:region[1]]
    # mask out background
    fuzzy_models_test_masked = dp.maskBKG(fuzzy_models_test,crisp_model_test)
    crisp_model_test         = crisp_model_test[...,np.newaxis].astype(np.float32)
    # model to parameter map
    print('tau (ms): '+str(tau_test))
    para_type    = 'time'
    imgs_gt_test = dp.model_to_T1w_image(sub_name=sub, model=fuzzy_models_test_masked,tau=tau_test,fraction=0.025,ac=8)
    maps_gt_test = dp.image2map(imgs=imgs_gt_test,tau=tau_test,fitting_model='T1_three_para_magn',signed=True,parameter_type=para_type,
                                algorithm='NLLS',pbar_disable=False,ac=1) # fitted map
    # save as tfrecord data
    imgs_gt_test = imgs_gt_test.astype(np.float32)
    maps_gt_test = maps_gt_test.astype(np.float32)
    tau_test     = np.repeat(np.reshape(tau_test,(1,-1)),repeats=imgs_gt_test.shape[0],axis=0).astype(np.float32)
    func.write2TFRecord_noise_free(img_gt=imgs_gt_test,map_gt=maps_gt_test,seg=crisp_model_test,tes=tau_test,\
                        filename=os.path.join(dir_data,'{}_tau_{}_sigma_{}_S_{}'.format(sub,0,sigma,num_slice)))
    print('-'*98)

#########################################################################################
print('='*98)
page_width = 7.16
fig,axes = plt.subplots(nrows=3,ncols=5,figsize=(page_width,page_width/5*3),dpi=600,tight_layout=True)
idx = 0
[ax.set_axis_off() for ax in axes.ravel()]
for i in range(5):
    axes[0,i].imshow(fuzzy_models_test_masked[idx,...,i],cmap='hot',vmin=0.0,vmax=1.0)
    axes[1,i].imshow(imgs_gt_test[idx,...,i],cmap='gray',vmin=0.0,vmax=1.2)
    axes[1,i].set_title('Noise free (TE'+str(i)+')')

axes[2,0].imshow(maps_gt_test[idx,...,0],cmap='gray',vmin=0.0,vmax=1.2) # A map
axes[2,1].imshow(maps_gt_test[idx,...,1],cmap='gray',vmin=0.0,vmax=2.4) # B map
axes[2,2].imshow(maps_gt_test[idx,...,2],cmap='jet',vmin=0.0,vmax=300.0) # T1* map (ms)
axes[2,3].imshow(1000.0/maps_gt_test[idx,...,2],cmap='jet',vmin=0.0,vmax=20.0) # R1* map (s-1)
axes[2,4].imshow(crisp_model_test[idx],cmap='jet') # tissue type
plt.savefig(os.path.join('figures','T1mapping','data_test'))

#########################################################################################
print('='*98)

