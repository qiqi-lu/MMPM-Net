import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib

import functions as func
import functions_show as func_show

##########################################################################################
sigma_test, type_te_test   = 0.05, 1
num_slice, num_realization = 3, 10

model      = 'admmnet'
name_model = 'admmnet_Ns_15_Nk_5_Nt_1_f_3_path_1_r2_maemow_6_mix_xfzz_tt_0.001_0.1'
# name_model = 'admmnet_Ns_15_Nk_5_Nt_1_f_3_path_1_r2_maemow_6_mix_xfzz_tt_0.001_0.1_dncnn_16_lr3'
# name_model = 'dopamine_Ns_10_path_1_SW_0_Norm_0_r2_maemow_6_mix_dxx'
# name_model = 'rim_Ns_6_f_36_r2_mmse_mean_6_mix_l2_0.001'
# name_model = 'resnet_4_moled_r2_mae_mono_6_mix'
# ----------------------------------------------------------------------------------------
id_slice_show   = 10
id_stage_show   = np.r_[0,1,3,5,10,14]
tick_stage_show = [0,1,3,5,10,15]
# ----------------------------------------------------------------------------------------
colume_width, page_width = 3.5, 7.16
font_dir = pathlib.Path(os.path.join('fonts','Helvetica.ttf'))
# ----------------------------------------------------------------------------------------
type_map_train, type_map_gt, type_map_show = 'R2', 'R2', 'R2'
sigma_train, type_te_train, scale = 'mix', 6, 100.0
flag_mask_out  = True

##########################################################################################
print('='*98)
print('Load results ...')
type_application = 'NLtst_{}_NLtrn_{}_TEtst_{}_TEtrn_{}_S_{}_N_{}'.\
                    format(sigma_test,sigma_train,type_te_test,type_te_train,num_slice,num_realization)
dir_results = os.path.join('results',type_application)

# ----------------------------------------------------------------------------------------
print('> ',dir_results)
maps_pred_m = np.load(os.path.join(dir_results,name_model,'maps_pred_'+model+'.npy'))
maps_gt     = np.load(os.path.join(dir_results,'maps_gt_test.npy'))
seg         = np.load(os.path.join(dir_results,'seg_test.npy'))

print('Results shape: ',maps_pred_m.shape)

maps_pred_m_recale = func_show.rescale_map(maps_pred_m,type_show=type_map_show,type_map=type_map_train,scale=scale)
maps_gt_rescale    = func_show.rescale_map(maps_gt,    type_show=type_map_show,type_map=type_map_gt,scale=scale)

# ----------------------------------------------------------------------------------------
# Region of Interest
ROI_wh = np.where(seg == 1,1,0)
for i in [2,3,8]: ROI_wh  = ROI_wh + np.where(seg == i,1,0)
ROI_brain = func.closing_hole(ROI_wh)
mask_bkg  = np.where(maps_gt_rescale > 0.0,1.0,0.0)

mask_out = np.repeat(ROI_brain,repeats=2,axis=-1)
# mask_out = mask_bkg

maps_pred_m_recale[1:] = maps_pred_m_recale[1:]*mask_out
maps_gt_rescale        = maps_gt_rescale*mask_out

##########################################################################################
# Evaluation Metrics
mappd = maps_pred_m_recale[...,1][...,np.newaxis]
mapgt = maps_gt_rescale[...,1][...,np.newaxis]
Ns    = maps_pred_m_recale.shape[0]
nrmse_p2, ssim_p2 = [], []

for i in range(Ns):
    nrmse_p2.append(func.nRMSE(y_pred=mappd[i],y_true=mapgt,roi=ROI_brain,average=False))
    ssim_p2.append( func.SSIM( y_pred=mappd[i],y_true=mapgt,roi=ROI_brain,average=False))
nrmse_p2 = np.stack(nrmse_p2)
ssim_p2  = np.stack(ssim_p2)
print(nrmse_p2.shape)

##########################################################################################
# Show each stage's output.
print('-'*98)
print('Show each stage output ...')
# ----------------------------------------------------------------------------------------
font_size        = 10.0
font_size_metrix = 8.0
font_size_tick   = 6.0
text_x, text_y   = 6, 42
# ----------------------------------------------------------------------------------------
max_map, max_error   = 20.0, 4.0
cmap_map, cmap_error = 'jet', 'jet'
N_stage_show = id_stage_show.shape[-1]
# ----------------------------------------------------------------------------------------
mappd = maps_pred_m_recale
mapgt = maps_gt_rescale
map_error = np.abs(mappd[...,1]-mapgt[...,1])
map_error = map_error*ROI_brain[...,0]
# ----------------------------------------------------------------------------------------
figure_width  = page_width
figure_heigth = page_width/(N_stage_show+1)*2*1.2
# ----------------------------------------------------------------------------------------
fig,axes = plt.subplots(nrows=2,ncols=N_stage_show+1,figsize=(figure_width,figure_heigth),dpi=600,constrained_layout=True)
[ax.set_axis_off() for ax in axes.ravel()]
# Ground Truth maps
axes[0,-1].imshow(mapgt[id_slice_show,:,:,1],cmap=cmap_map,vmin=0.0,vmax=max_map)
axes[0,-1].set_title('GT',font=font_dir,fontsize=font_size)
# Predict maps
for i,idx in enumerate(id_stage_show):
    pcm_map = axes[0,i].imshow(mappd[idx,id_slice_show,:,:,1],cmap=cmap_map,vmin=0.0,vmax=max_map)
    pcm_dif = axes[1,i].imshow(map_error[idx,id_slice_show],cmap=cmap_error,vmin=0.0,vmax=max_error)
    axes[0,i].set_title('Stage$_{{{}}}$'.format(tick_stage_show[i]),font=font_dir,fontsize=font_size)

for i,idx in enumerate(id_stage_show[1:]):
    axes[0,i+1].text(x=text_x,y=text_y,s=str(ssim_p2[idx,id_slice_show]),font=font_dir,fontsize=font_size_metrix,color='white')
    axes[1,i+1].text(x=text_x,y=text_y,s=str(nrmse_p2[idx,id_slice_show]),font=font_dir,fontsize=font_size_metrix,color='white')

cbar_map = fig.colorbar(pcm_map,ax=axes[0,:],shrink=0.8,aspect=24.0,ticks=[0,5,10,15,20])
cbar_dif = fig.colorbar(pcm_dif,ax=axes[1,:],shrink=0.8,aspect=24.0,ticks=[0,1,2,3,4])
cbar_map.ax.set_yticklabels(['0 s$^{-1}$',5,10,15,20],font=font_dir,fontdict={'fontsize':font_size_tick})
cbar_dif.ax.set_yticklabels(['0 s$^{-1}$',1,2,3,4],font=font_dir,fontdict={'fontsize':font_size_tick})
plt.savefig(os.path.join('figures','maps_'+model))

##########################################################################################
### small edition
id_stage_show = np.r_[0,1,3,15]
tick_stage_show = [0,1,3,15]
# ----------------------------------------------------------------------------------------
font_size        = 8
font_size_metrix = 8
font_size_tick   = 6
text_x, text_y   = 6, 44
# ----------------------------------------------------------------------------------------
max_map, max_error   = 20.0, 4.0
cmap_map, cmap_error = 'jet', 'jet'
N_stage_show = id_stage_show.shape[-1]
# ----------------------------------------------------------------------------------------
mappd = maps_pred_m_recale
mapgt = maps_gt_rescale
map_error = np.abs(mappd[...,1]-mapgt[...,1])
map_error = map_error*ROI_brain[...,0]
# ----------------------------------------------------------------------------------------
figure_width  = colume_width
figure_heigth = colume_width/N_stage_show*2*1.2
# ----------------------------------------------------------------------------------------
fig,axes = plt.subplots(nrows=2,ncols=N_stage_show,figsize=(figure_width,figure_heigth),dpi=600,constrained_layout=True)
[ax.set_axis_off() for ax in axes.ravel()]
# Predict maps
for i,idx in enumerate(id_stage_show):
    pcm_map = axes[0,i].imshow(mappd[idx,id_slice_show,:,:,1],cmap=cmap_map,vmin=0.0,vmax=max_map)
    pcm_dif = axes[1,i].imshow(map_error[idx,id_slice_show],cmap=cmap_error,vmin=0.0,vmax=max_error)
    axes[0,i].set_title('Stage$_{{{}}}$'.format(tick_stage_show[i]),font=font_dir,fontsize=font_size)

for i,idx in enumerate(id_stage_show[1:]):
    axes[0,i+1].text(x=text_x,y=text_y,s=str(ssim_p2[idx,id_slice_show]),font=font_dir,fontsize=font_size_metrix,color='white')
    axes[1,i+1].text(x=text_x,y=text_y,s=str(nrmse_p2[idx,id_slice_show]),font=font_dir,fontsize=font_size_metrix,color='white')

cbar_map = fig.colorbar(pcm_map,ax=axes[0,:],shrink=0.7,aspect=24.0,ticks=[0,5,10,15,20],location='left')
cbar_dif = fig.colorbar(pcm_dif,ax=axes[1,:],shrink=0.7,aspect=24.0,ticks=[0,1,2,3,4],location='left')
cbar_map.ax.set_yticklabels(['0',5,10,15,20],font=font_dir,fontdict={'fontsize':font_size_tick})
cbar_dif.ax.set_yticklabels(['0',1,2,3,4],font=font_dir,fontdict={'fontsize':font_size_tick})
plt.savefig(os.path.join('figures','maps_'+model+'_small'))
##########################################################################################
print('='*98)

