import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import numpy as np
import os
import pandas
import pathlib

import functions as func
import functions_show as func_show

##########################################################################################
sigma_test, type_te_test = 0.05, 1
num_slice, num_realization = 3, 10

name_network = ['admmnet_Ns_2_Nk_5_Nt_1_f_3_path_1_r2_maemow_6_mix_xfzz_tt_0.001_0.1',
                # 'admmnet_Ns_2_Nk_5_Nt_1_f_3_path_1_r2_maemow_6_mix_xfzz_tt_0.001_0.1_dncnn_16_lr3',
                'admmnet_Ns_5_Nk_5_Nt_1_f_3_path_1_r2_maemow_6_mix_xfzz_tt_0.001_0.1',
                'admmnet_Ns_10_Nk_5_Nt_1_f_3_path_1_r2_maemow_6_mix_xfzz_tt_0.001_0.1',
                # 'admmnet_Ns_10_Nk_5_Nt_1_f_3_path_1_r2_maemow_6_mix_xfzz_tt_0.001_0.1_dncnn_16_lr3',
                'admmnet_Ns_15_Nk_5_Nt_1_f_3_path_1_r2_maemow_6_mix_xfzz_tt_0.001_0.1',
                'admmnet_Ns_20_Nk_5_Nt_1_f_3_path_1_r2_maemow_6_mix_xfzz_tt_0.001_0.1']

name_model  = 'admmnet'

id_slice_show = 20

colume_width = 3.5  # inches
page_width   = 7.16 # inches
font_dir  = pathlib.Path(os.path.join('fonts','Helvetica.ttf'))

# -----------------------------------------------------------------------------------------
hyper_para = 'Ns'
name_hyper_para = 'N$_s$'
xlabel = 'Number of stage (N$_s$)'

# hyper_para = 'regul_w'
# name_hyper_para = '($\mu$,$\\rho$)'
# xlable = 'Number of stage ($N_s$)'

# -----------------------------------------------------------------------------------------
sigma_train, type_te_train, scale = 'mix', 6, 100.0
type_map_train, type_map_gt, type_map_show = 'R2','R2','R2'
type_application = 'NLtst_{}_NLtrn_{}_TEtst_{}_TEtrn_{}_S_{}_N_{}'\
                    .format(sigma_test,sigma_train,type_te_test,type_te_train,num_slice,num_realization)
dir_results = os.path.join('results',type_application)
N_network   = len(name_network)

##########################################################################################
print('='*98)
print('Load results ...')
print('> ',dir_results)
imgs_n  = np.load(os.path.join(dir_results,'imgs_n_test.npy'))
maps_gt = np.load(os.path.join(dir_results,'maps_gt_test.npy'))
seg     = np.load(os.path.join(dir_results,'seg_test.npy'))

maps_pred_m = []
for i in range(N_network):
    maps = np.load(os.path.join(dir_results,name_network[i],'maps_pred_{}.npy'.format(name_model)))
    maps_pred_m.append(maps)
    print(name_network[i],maps.shape)

# Rescale maps
maps_pred_m_recale = []
for i in range(N_network):
    maps_pred_m_recale.append(func_show.rescale_map(maps_pred_m[i], type_show=type_map_show, type_map=type_map_train, scale=scale))
maps_gt_rescale = func_show.rescale_map(maps_gt,type_show=type_map_show,type_map=type_map_gt,scale=scale)

# -----------------------------------------------------------------------------------------
ROI_wh = np.where(seg == 1,1,0)
for i in [2,3,8]: ROI_wh = ROI_wh + np.where(seg == i,1,0)
ROI_brain = func.closing_hole(ROI_wh)
mask_bkg  = np.where(maps_gt_rescale > 0.0,1.0,0.0)

# mask_out = mask_bkg
mask_out = np.repeat(ROI_brain,repeats=2,axis=-1)
# -----------------------------------------------------------------------------------------
maps_pred = [] # last output map
for i in range(N_network): 
    p = maps_pred_m_recale[i][-1]*mask_out
    maps_pred.append(p)
maps_pred = np.stack(maps_pred)
maps_gt_rescale = maps_gt_rescale*mask_out
print(maps_pred.shape)

##########################################################################################
# Evaluation Metrics
print('='*98)
print('Results evaluation (NRMSE and SSIM) ...')
# ----------------------------------------------------------------------------------------
mappd = maps_pred[...,1][...,np.newaxis]
mapgt = maps_gt_rescale[...,1][...,np.newaxis]
nrmse_p2, ssim_p2 = [],[]

for i in range(N_network):
    nrmse_p2.append(func.nRMSE(y_pred=mappd[i],y_true=mapgt,roi=ROI_brain,average=False))
    ssim_p2.append( func.SSIM( y_pred=mappd[i],y_true=mapgt,roi=ROI_brain,average=False))
nrmse_p2 = np.stack(nrmse_p2)
ssim_p2  = np.stack(ssim_p2)
print(nrmse_p2.shape)
# ----------------------------------------------------------------------------------------
# Print metrics table
print('-'*98)
name_metrics  = ['NRMSE(mean)','NRMSE(std)','SSIM(mean)','SSIM(std)']
data  = []

for i in range(N_network):
    data.append([np.mean(nrmse_p2[i]),np.std(nrmse_p2[i]),np.mean(ssim_p2[i]),np.std(ssim_p2[i])])
data  = np.round(np.array(data),decimals=5)
table = pandas.DataFrame(data,name_network,name_metrics)        
print(table)

##########################################################################################
# Maps
font_size        = 8 # 10, 8
font_size_metrix = 8 # 8, 6
font_size_tick   = 6
left_x, left_y   = 6, 44

xticklabels = [2,5,10,15,20]
# id_map_show = [0,1,2,3,4]
id_map_show = [0,1,3]

N_map_show = len(id_map_show)
vmax_map, vmax_error = 15, 1.5
# ----------------------------------------------------------------------------------------
### large edition
# grid_width    = page_width/8
# figure_width  = grid_width*6
# figure_heigth = grid_width*2.0*1.2
### small edition
grid_width    = colume_width/4
figure_width  = grid_width*4
figure_heigth = grid_width*2.0*1.2
# ----------------------------------------------------------------------------------------
fig,axes = plt.subplots(nrows=2,ncols=N_map_show+1,figsize=(figure_width,figure_heigth),dpi=600,constrained_layout=True)
[ax.set_axis_off() for ax in axes.ravel()]
# Ground Truth map
axes[0,0].set_title('GT',font=font_dir,fontsize=font_size)
axes[0,0].imshow(maps_gt_rescale[id_slice_show,...,1],cmap='jet',vmin=0.0,vmax=vmax_map)
# Predict maps
for i,idx in enumerate(id_map_show):
    axes[0,i+1].set_title(name_hyper_para+' = '+str(xticklabels[idx]),font=font_dir,fontsize=font_size)
    pcm_p = axes[0,i+1].imshow(maps_pred[i,id_slice_show,...,1],cmap='jet',vmin=0.0,vmax=vmax_map)
    emap  = np.abs(maps_pred[i,id_slice_show,...,1]-maps_gt_rescale[id_slice_show,...,1])*ROI_brain[id_slice_show,...,0]
    pcm_e = axes[1,i+1].imshow(emap,vmin=0.,vmax=vmax_error,cmap='jet')

    axes[0,i+1].text(x=left_x,y=left_y,s=str(ssim_p2[i,id_slice_show]), color='white',font=font_dir,fontsize=font_size_metrix)
    axes[1,i+1].text(x=left_x,y=left_y,s=str(nrmse_p2[i,id_slice_show]),color='white',font=font_dir,fontsize=font_size_metrix)

cbar_map=fig.colorbar(pcm_p,ax=axes[0,:],shrink=0.7,aspect=24.,location='left',ticks=[0,5,10,vmax_map])
cbar_dif=fig.colorbar(pcm_e,ax=axes[1,:],shrink=0.7,aspect=24.,location='left',ticks=[0,0.5,1.0,1.5,vmax_error])
cbar_map.ax.set_yticklabels([0,5,10,vmax_map],font=font_dir,fontdict={'fontsize':font_size_tick})
cbar_dif.ax.set_yticklabels([0,0.5,1,1.5,vmax_error],font=font_dir,fontdict={'fontsize':font_size_tick})
plt.savefig(os.path.join('figures','hyperpara_maps_'+hyper_para))

##########################################################################################
# NRMSE and SSIM profile
pos = range(N_network)
font_size      = 8   # 10, 8
font_size_tick = 6
line_width     = 0.5 # 0.5, 0.75
# ----------------------------------------------------------------------------------------
### large edition
# grid_width  = page_width/8
# figure_width  = grid_width*6
# figure_heigth = grid_width*2

### small edition
grid_width    = colume_width/3
figure_width  = grid_width*3
figure_heigth = grid_width*1
# ----------------------------------------------------------------------------------------
fig,axes = plt.subplots(nrows=1,ncols=2,figsize=(figure_width,figure_heigth),dpi=600,constrained_layout=True)
axes[0].errorbar(pos,np.mean(nrmse_p2[:,20:],axis=-1),np.std(nrmse_p2[:,20:],axis=-1),\
                linestyle='--',marker='o',capsize=2.0,color='blue',markersize=2.0,linewidth=line_width)
axes[0].set_ylabel('NRMSE',font=font_dir,fontsize=font_size)
axes[1].errorbar(pos,np.mean(ssim_p2[:,20:],axis=-1),np.std(ssim_p2[:,20:],axis=-1),\
                linestyle='--',marker='o',capsize=2.0,color='blue',markersize=2.0,linewidth=line_width)
axes[1].set_ylabel('SSIM',font=font_dir,fontsize=font_size)

for ax in axes.ravel(): 
    ax.set_xlabel(xlabel,font=font_dir,fontsize=font_size)
    ax.set_xticks(pos)
    ax.set_xticklabels(xticklabels,font=font_dir,size=font_size)
    ax.tick_params(axis='y', labelsize=font_size_tick,width=line_width,length=1.0,direction='in')
    ax.tick_params(axis='x', labelsize=font_size_tick,width=line_width,length=1.0,direction='in')
    plt.setp(ax.spines.values(), linewidth=line_width)
    ax.grid(axis='y',linewidth=line_width)

plt.savefig(os.path.join('figures','hyperpara_profile_'+hyper_para))
print('='*98)
##########################################################################################
