import pathlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas
import math

import functions as func
import functions_show as func_show

##########################################################################################
# 1，4，7，13
sigma_test, type_te_test   = 0.05, 13

name_resnet   = 'resnet_4_moled_r2_mae_mono_6_mix_10'
name_rim      = 'rim_Ns_6_f_36_r2_mmse_mean_6_mix_l2_0.001'
name_dopamine = 'dopamine_Ns_10_path_1_SW_0_Norm_0_r2_maemow_6_mix_dxx_11'
name_admmnet  = 'admmnet_Ns_15_Nk_5_Nt_1_f_3_path_1_r2_maemow_6_mix_xfzz_tt_0.001_0.1_11'
# name_admmnet  = 'admmnet_Ns_15_Nk_5_Nt_1_f_3_path_1_r2_maemow_6_mix_xfzz_tt_0.001_0.1_dncnn_10_lr3'

num_slice, num_realization = 1, 10
id_slice_show   = 0
# ----------------------------------------------------------------------------------------
id_methods_show = [0,1,2,3,4]
id_rois = [1,2,3,8]

method = ['MLE','ResNet','RIM','DOPAMINE','MMPM-Net']
colors = ['purple','black','green','blue','red']
# ----------------------------------------------------------------------------------------
type_map_train_admmnet, type_map_train_dopamine, type_map_train_rim, type_map_train_resnet, type_map_mle = 'R2','R2','R2','R2','R2'
type_map_gt, type_map_show = 'R2', 'R2'
sigma_train, type_te_train, scale = 'mix', 6, 100.0

type_test = 'wilcoxon'
# type_test = 'student'
# type_test = 'student_p'

colume_width = 3.5
page_width   = 7.16
font_dir  = pathlib.Path(os.path.join('fonts','Helvetica.ttf'))

##########################################################################################
print('='*98)
print('Load results ...')
type_application = 'NLtst_{}_NLtrn_{}_TEtst_{}_TEtrn_{}_S_{}_N_{}'.format(sigma_test,sigma_train,type_te_test,type_te_train,num_slice,num_realization)
dir_results = os.path.join('results',type_application)

# ----------------------------------------------------------------------------------------
print('> ',dir_results)
imgs_n = np.load(os.path.join(dir_results,'imgs_n_test.npy'))
maps_mle             = np.load(os.path.join(dir_results,'maps_mle.npy'))
maps_pred_resnet     = np.load(os.path.join(dir_results,name_resnet,'maps_pred_resnet.npy'))
maps_pred_rim_m      = np.load(os.path.join(dir_results,name_rim,'maps_pred_rim.npy'))
maps_pred_dopamine_m = np.load(os.path.join(dir_results,name_dopamine,'maps_pred_dopamine.npy'))
maps_pred_admmnet_m  = np.load(os.path.join(dir_results,name_admmnet,'maps_pred_admmnet.npy'))
maps_gt  = np.load(os.path.join(dir_results,'maps_gt_test.npy'))
seg = np.load(os.path.join(dir_results,'seg_test.npy'))

print(maps_pred_resnet.shape,name_resnet)
print(maps_pred_rim_m.shape,name_rim)
print(maps_pred_dopamine_m.shape,name_dopamine)
print(maps_pred_admmnet_m.shape,name_admmnet)

maps_mle_rescale            = func_show.rescale_map(maps_mle,            type_show=type_map_show,type_map=type_map_mle,           scale=scale)
maps_pred_resnet_recale     = func_show.rescale_map(maps_pred_resnet,    type_show=type_map_show,type_map=type_map_train_resnet,  scale=scale)
maps_pred_rim_m_recale      = func_show.rescale_map(maps_pred_rim_m,     type_show=type_map_show,type_map=type_map_train_rim,     scale=scale)
maps_pred_dopamine_m_recale = func_show.rescale_map(maps_pred_dopamine_m,type_show=type_map_show,type_map=type_map_train_dopamine,scale=scale)
maps_pred_admmnet_m_recale  = func_show.rescale_map(maps_pred_admmnet_m, type_show=type_map_show,type_map=type_map_train_admmnet, scale=scale)
maps_gt_rescale             = func_show.rescale_map(maps_gt,             type_show=type_map_show,type_map=type_map_gt,            scale=scale)

# ----------------------------------------------------------------------------------------
ROI_wh = np.where(seg == 1,1,0)
for i in [2,3,8]: ROI_wh  = ROI_wh + np.where(seg == i,1,0)
ROI_brain = func.closing_hole(ROI_wh)
mask_bkg  = np.where(maps_gt_rescale > 0.0,1.0,0.0)

ROIs = []
for i in id_rois: ROIs.append(np.where(seg == i,1,0))

# mask_out = mask_bkg
mask_out = np.repeat(ROI_brain,repeats=2,axis=-1)
# ----------------------------------------------------------------------------------------
# Evaluation on last output maps and mask out background
maps_pred = []
maps_pred.append(maps_mle_rescale)
maps_pred.append(maps_pred_resnet_recale)
maps_pred.append(maps_pred_rim_m_recale[-1])
maps_pred.append(maps_pred_dopamine_m_recale[-1])
maps_pred.append(maps_pred_admmnet_m_recale[-1])

N_methods = len(maps_pred)
for i in range(N_methods): maps_pred[i] = maps_pred[i]*mask_out
maps_gt_rescale = maps_gt_rescale*mask_out
##########################################################################################
print('='*98)
print('Evaluation results (NRMSE and SSIM) ...')
nrmse_m0, nrmse_p2, ssim_m0, ssim_p2 = [],[],[],[]
for maps in maps_pred:
    nrmse_m0.append(func.nRMSE(y_pred=maps[...,0][...,np.newaxis],y_true=maps_gt_rescale[...,0][...,np.newaxis],roi=ROI_brain,average=False))
    nrmse_p2.append(func.nRMSE(y_pred=maps[...,1][...,np.newaxis],y_true=maps_gt_rescale[...,1][...,np.newaxis],roi=ROI_brain,average=False))
    ssim_m0.append( func.SSIM( y_pred=maps[...,0][...,np.newaxis],y_true=maps_gt_rescale[...,0][...,np.newaxis],roi=ROI_brain,average=False))
    ssim_p2.append( func.SSIM( y_pred=maps[...,1][...,np.newaxis],y_true=maps_gt_rescale[...,1][...,np.newaxis],roi=ROI_brain,average=False))

title = ['NRMSE(mean)','NRMSE(std)','SSIM(mean)','SSIM(std)']
data_m0, data_r2 = [],[]
for i in range(N_methods): 
    data_m0.append([np.mean(nrmse_m0[i]),np.std(nrmse_m0[i]),np.mean(ssim_m0[i]),np.std(ssim_m0[i])])
    data_r2.append([np.mean(nrmse_p2[i]),np.std(nrmse_p2[i]),np.mean(ssim_p2[i]),np.std(ssim_p2[i])])

data_m0  = np.round(np.array(data_m0),decimals=5)
data_r2  = np.round(np.array(data_r2),decimals=5)

table_m0 = pandas.DataFrame(data_m0,method,title)
table_r2 = pandas.DataFrame(data_r2,method,title)

print('-'*98)
# print(table_m0)
print('-'*98)
print('R2')
print(table_r2)
# ----------------------------------------------------------------------------------------
# P value
idx_ref = 4
idx_com = [0,1,2,3]

nrmse_m0_p, nrmse_p2_p, ssim_m0_p,ssim_p2_p = [],[],[],[]
for i in idx_com:
    nrmse_m0_p.append(func_show.significance_test(nrmse_m0[idx_ref],nrmse_m0[i],type=type_test,equal_var='False',return_statistic=False))
    nrmse_p2_p.append(func_show.significance_test(nrmse_p2[idx_ref],nrmse_p2[i],type=type_test,equal_var='False',return_statistic=False))
    ssim_m0_p.append( func_show.significance_test(ssim_m0[idx_ref], ssim_m0[i], type=type_test,equal_var='False',return_statistic=False))
    ssim_p2_p.append( func_show.significance_test(ssim_p2[idx_ref], ssim_p2[i], type=type_test,equal_var='False',return_statistic=False))

title_vs  = ['NRMSE(M0)','NRMSE(R2*)','SSIM(M0)','SSIM(R2*)']
method_vs, data_p = [],[]

for i in idx_com: 
    method_vs.append(method[idx_ref]+' vs '+method[i])
for i in range(len(idx_com)): 
    data_p.append([nrmse_m0_p[i],nrmse_p2_p[i],ssim_m0_p[i],ssim_p2_p[i]])
data_p  = np.array(data_p)
table_p = pandas.DataFrame(data_p,method_vs,title_vs)

print('-'*98)
print(table_p)

fig,axes=plt.subplots(nrows=1,ncols=2,dpi=300,figsize=(colume_width,colume_width*0.5),constrained_layout=True)
for i in [1,2,3,4]:
    axes[0].hist(x=nrmse_p2[i],bins=10, histtype='step', stacked=True, fill=False)
    axes[1].hist(x=ssim_p2[i],bins=10, histtype='step', stacked=True, fill=False)
plt.savefig(os.path.join('figures','metrix_hist'))

# ----------------------------------------------------------------------------------------
# Significance
data_significance  = np.where(data_p<0.001,'***',' ')
data_significance  = np.where((data_p<0.01)&(data_p>=0.001),'**',data_significance)
data_significance  = np.where((data_p<0.05)&(data_p>=0.01),'*',data_significance)
table_significance = pandas.DataFrame(data_significance,method_vs,title_vs)   

print('-'*98)
print(table_significance)

##########################################################################################
print('='*98)
print('ROI analysis ...')
labels_rois = ['CSF','White Matter','Gray Matter','Vessel']
colors_rois = ['red','blue','green','gray']
N_roi  = len(id_rois)
N_methods_show = len(id_methods_show)

idx_pixel_rois = []
for x in ROIs: idx_pixel_rois.append(np.where(x[id_slice_show,...,-1])) 

p2_rois_gt = []
for idx in idx_pixel_rois:
    p2_rois_gt.append(maps_gt_rescale[id_slice_show,...,1][idx])

p2_rois_methods = []
for maps in maps_pred:
    p2_rois = []
    for idx in idx_pixel_rois: p2_rois.append(maps[id_slice_show,...,1][idx])
    p2_rois_methods.append(p2_rois)

# ----------------------------------------------------------------------------------------
font_size = 10.0
font_size_legend = 6.0
font_size_tick   = 6.0
box_line_width = 0.5

xlim,ylim = (0.0,16.0),(0.0,16.0)
Nrows = int(math.ceil(N_methods_show/3))

# ----------------------------------------------------------------------------------------
fig,axes  = plt.subplots(nrows=Nrows,ncols=3,dpi=600,figsize=(page_width,page_width/3.0*Nrows),tight_layout=True)
axes = axes.ravel()

for i in range(N_methods_show):
    axes[i].plot(xlim,ylim,'--',color='black',linewidth=box_line_width)
    for j in range(N_roi): 
        axes[i].plot(p2_rois_gt[j],p2_rois_methods[i][j],'o'\
            ,markersize=0.2,alpha=0.2,label=labels_rois[j],markeredgecolor=colors_rois[j],markerfacecolor='none')
    axes[i].set_ylabel('$\mathrm{R_2}$'+' ('+method[id_methods_show[i]]+') '+'('+'$\mathrm{s^{-1}}$'+')',font=font_dir,fontsize=font_size)
    axes[i].set_xlabel('$\mathrm{R_2}$ (Reference) ($\mathrm{s^{-1}}$)',font=font_dir,fontsize=font_size)
axes[0].legend(markerscale=15.0,prop={'size':font_size_legend})
axes[-1].set_axis_off()

for ax in axes:
    ax.tick_params(axis='y', labelsize=font_size_tick,width=box_line_width,length=2.0)
    ax.tick_params(axis='x', labelsize=font_size_tick,width=box_line_width,length=2.0)
    ax.set_xlim(xlim),ax.set_ylim(ylim)
    plt.setp(ax.spines.values(), linewidth=box_line_width)

plt.savefig('figures/roi_analysis')

##########################################################################################
# Show maps
print('='*98)
print('Show maps ...')
font_size = 8
font_size_metrix = 7
font_size_legend = 6
left_x, left_y   = 6, 42

max_map = 20.0
error_range = [0.0,2.0]
cmap_map, s_map = 'jet','$\mathrm{R_2}$'
para_ssim, para_nrmse = ssim_p2, nrmse_p2

cmap_error = 'jet'
error_map = []

for i in range(N_methods): 
    error_map.append(np.abs(maps_pred[i]-maps_gt_rescale))

for i in range(N_methods): 
    error_map[i] = error_map[i]*np.repeat(ROI_brain,repeats=2,axis=-1)

# --------------------------------------------------------------------------------------------------------
# figure with ResNet and Ground Truth maps. (transverse)
# --------------------------------------------------------------------------------------------------------
grid_width    = page_width/8
figure_width  = grid_width*6
figure_heigth = grid_width*2.0*1.12
# --------------------------------------------------------------------------------------------------------
fig, axes = plt.subplots(nrows=2,ncols=N_methods_show+1,figsize=(figure_width,figure_heigth),dpi=600,constrained_layout=True)
[ax.set_axis_off() for ax in axes.ravel()]
# Ground Truth maps
pcm_map = axes[0,0].imshow(maps_gt_rescale[id_slice_show,:,:,1],cmap=cmap_map,vmin=0.0,vmax=max_map)
axes[0,0].text(x=left_x,y=left_y,s=s_map,color='white',font=font_dir,fontsize=font_size_metrix)
axes[0,0].set_title('GT',font=font_dir,fontsize=font_size)
# Predict maps
for i in id_methods_show:
    axes[0,i+1].imshow(maps_pred[i][id_slice_show,:,:,1],cmap=cmap_map,vmin=0.0,vmax=max_map)
    axes[0,i+1].text(x=left_x,y=left_y,s=str(para_ssim[i][id_slice_show]),color='white',font=font_dir,fontsize=font_size_metrix)
    axes[0,i+1].set_title(method[i],font=font_dir,fontsize=font_size)

    pcm_dif = axes[1,i+1].imshow(error_map[i][id_slice_show,:,:,1],cmap=cmap_error,vmin=error_range[0],vmax=error_range[1])
    axes[1,i+1].text(x=left_x,y=left_y,s=str(para_nrmse[i][id_slice_show]),color='white',font=font_dir,fontsize=font_size_metrix)

cbar_map = fig.colorbar(pcm_map,ax=axes[0,:],shrink=0.8,aspect=30,location='right',ticks=[max_map*x/4.0 for x in range(5)])
cbar_dif = fig.colorbar(pcm_dif,ax=axes[1,:],shrink=0.8,aspect=30,location='right',ticks=[error_range[-1]*x/4.0 for x in range(5)])
cbar_map.ax.set_yticklabels(['0 $s^{-1}$','5','10','15','20'],font=font_dir,fontdict={'fontsize':font_size_legend})
cbar_dif.ax.set_yticklabels(['0 $s^{-1}$','0.5','1','1.5','2'],font=font_dir,fontdict={'fontsize':font_size_legend})
plt.savefig(os.path.join('figures','maps_predict_simu_a'))
# --------------------------------------------------------------------------------------------------------
# figure with ResNet and Ground Truth maps. (longitudinal)
# --------------------------------------------------------------------------------------------------------
grid_width    = colume_width/4
figure_width  = grid_width*2
figure_heigth = grid_width*(N_methods_show+1)*1.2
# --------------------------------------------------------------------------------------------------------
fig, axes = plt.subplots(nrows=N_methods_show+1,ncols=2,figsize=(figure_width,figure_heigth),dpi=600,constrained_layout=True)
[ax.set_axis_off() for ax in axes.ravel()]
# Ground Truth maps
pcm_map = axes[0,0].imshow(maps_gt_rescale[id_slice_show,:,:,1],cmap=cmap_map,vmin=0.0,vmax=max_map)
axes[0,0].text(x=left_x,y=left_y,s=s_map,color='white',font=font_dir,fontsize=font_size_metrix)
# Predict maps
for i in id_methods_show:
    # map
    axes[i+1,0].imshow(maps_pred[i][id_slice_show,:,:,1],cmap=cmap_map,vmin=0.0,vmax=max_map)
    axes[i+1,0].text(x=left_x,y=left_y,s=str(para_ssim[i][id_slice_show]),color='white',font=font_dir,fontsize=font_size_metrix)
    # absolute difference map
    pcm_dif = axes[i+1,1].imshow(error_map[i][id_slice_show,:,:,1],cmap=cmap_error,vmin=error_range[0],vmax=error_range[1])
    axes[i+1,1].text(x=left_x,y=left_y,s=str(para_nrmse[i][id_slice_show]),color='white',font=font_dir,fontsize=font_size_metrix)
# color bar
cbar_map = fig.colorbar(pcm_map,ax=axes[:,0],shrink=0.7,aspect=24,location='bottom',ticks=[max_map*x/4.0 for x in range(5)])
cbar_dif = fig.colorbar(pcm_dif,ax=axes[:,1],shrink=0.7,aspect=24,location='bottom',ticks=[error_range[-1]*x/4.0 for x in range(5)])
cbar_map.ax.set_xticklabels(['0 ','5','10','15','20'],font=font_dir,fontdict={'fontsize':font_size_legend})
cbar_dif.ax.set_xticklabels(['0 ','0.5','1','1.5','2'],font=font_dir,fontdict={'fontsize':font_size_legend})
plt.savefig(os.path.join('figures','maps_predict_simu_a_longutidinal'))

# --------------------------------------------------------------------------------------------------------
# figure without Resnet and Ground Truth maps.
# --------------------------------------------------------------------------------------------------------
grid_width    = colume_width/4
figure_width  = grid_width*4
figure_heigth = grid_width*2.0*1.12
# --------------------------------------------------------------------------------------------------------
fig, axes = plt.subplots(nrows=2,ncols=N_methods_show-1,figsize=(figure_width,figure_heigth),dpi=600,constrained_layout=True)
[ax.set_axis_off() for ax in axes.ravel()]
# Predict maps
for i,idx in enumerate([0,2,3,4]):
    pcm_map = axes[0,i].imshow(maps_pred[idx][id_slice_show,:,:,1],cmap=cmap_map,vmin=0.0,vmax=max_map)
    axes[0,i].text(x=left_x,y=left_y,s=str(para_ssim[idx][id_slice_show]),color='white',font=font_dir,fontsize=font_size_metrix)
    axes[0,i].set_title(method[idx],font=font_dir,fontsize=font_size)

    pcm_dif = axes[1,i].imshow(error_map[idx][id_slice_show,:,:,1],cmap=cmap_error,vmin=error_range[0],vmax=error_range[1])
    axes[1,i].text(x=left_x,y=left_y,s=str(para_nrmse[idx][id_slice_show]),color='white',font=font_dir,fontsize=font_size_metrix)

cbar_map = fig.colorbar(pcm_map,ax=axes[0,:],shrink=0.8,aspect=30,location='right',ticks=[max_map*x/4.0 for x in range(5)])
cbar_dif = fig.colorbar(pcm_dif,ax=axes[1,:],shrink=0.8,aspect=30,location='right',ticks=[error_range[-1]*x/4.0 for x in range(5)])
cbar_map.ax.set_yticklabels(['0 $s^{-1}$','5','10','15','20'],font=font_dir,fontdict={'fontsize':font_size_legend})
cbar_dif.ax.set_yticklabels(['0 $s^{-1}$','0.5','1','1.5','2'],font=font_dir,fontdict={'fontsize':font_size_legend})
plt.savefig(os.path.join('figures','maps_predict_simu_b'))

# --------------------------------------------------------------------------------------------------------
# figure without Resnet but with ground truth.
# --------------------------------------------------------------------------------------------------------
grid_width    = page_width/8
figure_width  = grid_width*5
figure_heigth = grid_width*2.0*1.12

fig, axes = plt.subplots(nrows=2,ncols=N_methods_show,figsize=(figure_width,figure_heigth),dpi=600,constrained_layout=True)
[ax.set_axis_off() for ax in axes.ravel()]
# Ground Truth maps
pcm_map = axes[0,0].imshow(maps_gt_rescale[id_slice_show,:,:,1],cmap=cmap_map,vmin=0.0,vmax=max_map)
axes[0,0].text(x=left_x,y=left_y,s=s_map,color='white',font=font_dir,fontsize=font_size_metrix)
axes[0,0].set_title('GT',font=font_dir,fontsize=font_size)
# Predict maps
for i,idx in enumerate([0,2,3,4]):
    pcm_map = axes[0,i+1].imshow(maps_pred[idx][id_slice_show,:,:,1],cmap=cmap_map,vmin=0.0,vmax=max_map)
    axes[0,i+1].text(x=left_x,y=left_y,s=str(para_ssim[idx][id_slice_show]),color='white',font=font_dir,fontsize=font_size_metrix)
    axes[0,i+1].set_title(method[idx],font=font_dir,fontsize=font_size)

    pcm_dif = axes[1,i+1].imshow(error_map[idx][id_slice_show,:,:,1],cmap=cmap_error,vmin=error_range[0],vmax=error_range[1])
    axes[1,i+1].text(x=left_x,y=left_y,s=str(para_nrmse[idx][id_slice_show]),color='white',font=font_dir,fontsize=font_size_metrix)

cbar_map = fig.colorbar(pcm_map,ax=axes[0,:],shrink=0.8,aspect=30,location='right',ticks=[max_map*x/4.0 for x in range(5)])
cbar_dif = fig.colorbar(pcm_dif,ax=axes[1,:],shrink=0.8,aspect=30,location='right',ticks=[error_range[-1]*x/4.0 for x in range(5)])
cbar_map.ax.set_yticklabels(['0 $s^{-1}$','5','10','15','20'],font=font_dir,fontdict={'fontsize':font_size_legend})
cbar_dif.ax.set_yticklabels(['0 $s^{-1}$','0.5','1','1.5','2'],font=font_dir,fontdict={'fontsize':font_size_legend})
plt.savefig(os.path.join('figures','maps_predict_simu_c'))

##########################################################################################
# NRMSE and SSIM values of different methods as the number of the stage (s) increased.
print('-'*98)
metrics_ms = []
metrics_ms.append(func_show.evaluate_mo(maps=maps_mle_rescale,maps_gt=maps_gt_rescale,mask=ROI_brain))
metrics_ms.append(func_show.evaluate_mo(maps=maps_pred_resnet_recale,maps_gt=maps_gt_rescale,mask=ROI_brain))
metrics_ms.append(func_show.evaluate_mo(maps=maps_pred_rim_m_recale,maps_gt=maps_gt_rescale,mask=ROI_brain))
metrics_ms.append(func_show.evaluate_mo(maps=maps_pred_dopamine_m_recale,maps_gt=maps_gt_rescale,mask=ROI_brain))
metrics_ms.append(func_show.evaluate_mo(maps=maps_pred_admmnet_m_recale,maps_gt=maps_gt_rescale,mask=ROI_brain))

# --------------------------------------------------------------------------------------------------------
Ns_max = np.max([m.shape[-1] for m in metrics_ms])
nrmse_limit = [0.0,0.25]
ssim_limit  = [0.5,1.0]

# --------------------------------------------------------------------------------------------------------
font_size        = 10.0
font_size_tick   = 6.0
font_size_legend = 5.0
box_line_width   = 0.5
marker_size      = 2.0
# --------------------------------------------------------------------------------------------------------
fig,axes = plt.subplots(nrows=1,ncols=2,figsize=(colume_width,colume_width*0.5),dpi=300,constrained_layout=True)

for m, n in enumerate([1,3]):
    axes[m].plot([0.0, Ns_max-1],[metrics_ms[0][n],metrics_ms[0][n]],'--',color='gray',linewidth=box_line_width)
    axes[m].plot(0.0,metrics_ms[0][n],'.',markersize=marker_size,color=colors[0],label=method[0],linewidth=box_line_width)

    if 1 in id_methods_show:
        axes[m].plot([0.0, Ns_max-1],[metrics_ms[1][n],metrics_ms[1][n]],'--',color='gray',linewidth=box_line_width)
        axes[m].plot(0.0,metrics_ms[1][n],'.',markersize=marker_size,color=colors[1],label=method[1],linewidth=box_line_width)

    for i in [2,3,4]: 
        axes[m].plot(metrics_ms[i][n],'.-',markersize=marker_size,color=colors[i],label=method[i],linewidth=box_line_width)

axes[0].set_ylabel('NRMSE',font=font_dir,fontsize=font_size)
axes[0].set_ylim(nrmse_limit)
axes[0].legend(prop={'size':font_size_legend})

axes[1].set_ylabel('SSIM',font=font_dir,fontsize=font_size)
axes[1].set_ylim(ssim_limit)

for ax in axes.ravel(): 
    ax.set_xlabel('Stage',font=font_dir,fontsize=font_size)
    plt.setp(ax.spines.values(), linewidth=box_line_width)
    ticks = np.arange(start=0,stop=Ns_max,step=2)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks,fontdict={'fontsize':font_size_tick})
    ax.tick_params(axis='y', labelsize=font_size_tick,width=box_line_width,length=2.0)
    ax.tick_params(axis='x', labelsize=font_size_tick,width=box_line_width,length=2.0)

plt.savefig(os.path.join('figures','evaluation_stage'))

# --------------------------------------------------------------------------------------------------------
# evaluation stage of ADMM-REDNet
metri_admmnet = func_show.evaluate_mo(maps=maps_pred_admmnet_m_recale,maps_gt=maps_gt_rescale,mask=ROI_brain,ave=False)
metri_admmnet_mean = np.mean(metri_admmnet,axis=-1)
metri_admmnet_std  = np.std(metri_admmnet,axis=-1)
### small edition
Ns_max = 16
nrmse_limit = [0.0,0.25]
ssim_limit  = [0.5,1.0]
# --------------------------------------------------------------------------------------------------------
font_size        = 8.0
font_size_tick   = 6.0
line_width       = 0.5
marker_size      = 2.0
# --------------------------------------------------------------------------------------------------------
grid_width    = colume_width/3
figure_width  = grid_width*3
figure_heigth = grid_width*1
# --------------------------------------------------------------------------------------------------------
fig,axes = plt.subplots(nrows=1,ncols=2,figsize=(figure_width,figure_heigth),dpi=600,constrained_layout=True)
i = 4 # admm-rednet
for m, n in enumerate([1,3]):
    axes[m].plot(metri_admmnet_mean[n],'.-',markersize=marker_size,color=colors[i],label=method[i],linewidth=box_line_width)
    # axes[m].errorbar(range(Ns_max),metri_admmnet_mean[n],yerr=metri_admmnet_std[n],\
        # linestyle='-',marker='o',capsize=marker_size,color='red',markersize=marker_size,linewidth=line_width)
axes[0].set_ylabel('NRMSE',font=font_dir,fontsize=font_size)
axes[0].set_ylim(nrmse_limit)
axes[1].set_ylabel('SSIM',font=font_dir,fontsize=font_size)
axes[1].set_ylim(ssim_limit)

for ax in axes.ravel(): 
    ticks = np.arange(start=0,stop=Ns_max,step=2)
    ax.set_xlabel('Stage Index',font=font_dir,fontsize=font_size)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks,fontdict={'fontsize':font_size_tick})
    ax.tick_params(axis='y', labelsize=font_size_tick,width=line_width,length=1.0,direction='in')
    ax.tick_params(axis='x', labelsize=font_size_tick,width=line_width,length=1.0,direction='in')
    plt.setp(ax.spines.values(), linewidth=line_width)
    ax.grid(axis='y',linewidth=line_width)

plt.savefig(os.path.join('figures','evaluation_stage_admmrednet'))

##########################################################################################
print('='*98)