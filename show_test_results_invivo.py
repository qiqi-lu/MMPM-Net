import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import pathlib
import scipy

import functions_show as func_show

##########################################################################################
type_te_test  = 13

name_resnet   = 'resnet_4_moled_r2_mae_mono_6_mix'
name_rim      = 'rim_Ns_6_f_36_r2_mmse_mean_6_mix_l2_0.001'
name_dopamine = 'dopamine_Ns_10_path_1_SW_0_Norm_0_r2_maemow_6_mix_dxx_11'
name_admmnet  = 'admmnet_Ns_15_Nk_5_Nt_1_f_3_path_1_r2_maemow_6_mix_xfzz_tt_0.001_0.1_11'
# name_admmnet  = 'admmnet_Ns_10_Nk_5_Nt_1_f_3_path_1_r2_maemow_6_mix_xfzz_tt_0.001_0.1_dncnn_16_lr3'

name_data     = 'invivo'
name_study    = ['Study_30_1']
name_protocol = 'MESE'

method = ['MLE','ResNet','RIM','DOPAMINE','MMPM-Net']
id_methods_show = [0,2,3,4]
id_slice_show   = 0
id_repeat_show  = 0
flag_mask_out   = 1
# ----------------------------------------------------------------------------------------
sigma_train, type_te_train, scale = 'mix', 6, 100.0
type_map_train_admmnet, type_map_train_dopamine, type_map_train_rim, type_map_train_resnet, type_map_mle = 'R2','R2','R2','R2','R2'
type_map_show = 'R2'

colume_width = 3.5
page_width   = 7.16
font_dir     = pathlib.Path(os.path.join('fonts','Helvetica.ttf'))

##########################################################################################
print('='*98)
print('Load results ...')
imgs_n, maps_pred_admmnet_m, maps_pred_dopamine_m, maps_pred_rim_m, maps_pred_resnet, maps_mle = [],[],[],[],[],[]
for name in name_study:
    type_application = '{}_{}_{}_NLtrn_{}_TEtst_{}_TEtrn_{}'.format(name_data,name,name_protocol,sigma_train,type_te_test,type_te_train)
    dir_results      = os.path.join('results',type_application)
    print('> ',dir_results)
    imgs_n.append(              np.load(os.path.join(dir_results,'imgs_n_test.npy')))
    maps_mle.append(            np.load(os.path.join(dir_results,'maps_mle.npy')))
    maps_pred_resnet.append(    np.load(os.path.join(dir_results,name_resnet,'maps_pred_resnet.npy')))
    maps_pred_rim_m.append(     np.load(os.path.join(dir_results,name_rim,'maps_pred_rim.npy')))
    maps_pred_dopamine_m.append(np.load(os.path.join(dir_results,name_dopamine,'maps_pred_dopamine.npy')))
    maps_pred_admmnet_m.append( np.load(os.path.join(dir_results,name_admmnet,'maps_pred_admmnet.npy')))

imgs_n   = np.stack(imgs_n)
maps_mle = np.stack(maps_mle)
maps_pred_resnet     = np.stack(maps_pred_resnet)
maps_pred_rim_m      = np.stack(maps_pred_rim_m)
maps_pred_dopamine_m = np.stack(maps_pred_dopamine_m)
maps_pred_admmnet_m  = np.stack(maps_pred_admmnet_m)

# ----------------------------------------------------------------------------------------
# Rescale maps
maps_mle_rescale            = func_show.rescale_map(maps_mle,            type_show=type_map_show,type_map=type_map_mle,            scale=scale)
maps_pred_resnet_recale     = func_show.rescale_map(maps_pred_resnet,    type_show=type_map_show,type_map=type_map_train_resnet,   scale=scale)
maps_pred_rim_m_recale      = func_show.rescale_map(maps_pred_rim_m,     type_show=type_map_show,type_map=type_map_train_rim,      scale=scale)
maps_pred_dopamine_m_recale = func_show.rescale_map(maps_pred_dopamine_m,type_show=type_map_show,type_map=type_map_train_dopamine, scale=scale)
maps_pred_admmnet_m_recale  = func_show.rescale_map(maps_pred_admmnet_m, type_show=type_map_show,type_map=type_map_train_admmnet,  scale=scale)

print('Image shape: ',imgs_n.shape)
print('Map shape (ADMM-REDNet): ',maps_pred_admmnet_m.shape)

# ----------------------------------------------------------------------------------------
# Evaluation on last output maps
print('='*98)
maps_pred = []
maps_pred.append(maps_mle_rescale)
maps_pred.append(maps_pred_resnet_recale)
maps_pred.append(maps_pred_rim_m_recale[:,-1])
maps_pred.append(maps_pred_dopamine_m_recale[:,-1])
maps_pred.append(maps_pred_admmnet_m_recale[:,-1])

maps_pred = np.stack(maps_pred)
print('(N_methods,N_repeat,N_slices,Ny,Nx,Nz) = ',maps_pred.shape)

N_methods,N_repeat,N_slices = maps_pred.shape[0:3]
maps_pred_nomask = maps_pred
# Mask out background
if flag_mask_out == 1:
    for i,name in enumerate(name_study):
        mask_bkg = np.load(file=os.path.join('data',name_data+'data',name,'ROI.npy'))
        mask_bkg = np.repeat(np.abs(mask_bkg)[...,np.newaxis],repeats=2,axis=-1)
        for j in range(N_methods): maps_pred[j,i] = maps_pred[j,i]*mask_bkg

##########################################################################################
print('='*98)
print('ROI Analsis ...')
r2_pix = []
for i_study, name in enumerate(name_study):
    mask_roi = np.load(file=os.path.join('data',name_data+'data',name,'ROI.npy'))
    mask_roi = np.abs(mask_roi)
    N_mask   = mask_roi.shape[0]

    m = []
    for i_method in range(N_methods):
        a = []
        for i_slice in range(N_slices):
            b = []
            for i_mask in range(N_mask):
                pix_idx = np.where(mask_roi[i_mask])
                pix     = np.nan_to_num(maps_pred[i_method,i_study,i_slice,...,1][pix_idx],nan=0.0,posinf=0.0,neginf=0.0)
                b.append(pix)
            a.append(b)
        m.append(a)
    r2_pix.append(m)

r2_pix = np.transpose(np.array(r2_pix),axes=(1,0,2,3,4))
print(r2_pix.shape)

##########################################################################################
# Predict maps
y_box, x_box, size_box = 100, 100, 100

font_size      = 6 # 10, 6
font_size_tick = 6
line_width     = 0.5

N_methods_show = len(id_methods_show)
map_max = 20.0
# ----------------------------------------------------------------------------------------
### large edition
# grid_width    = page_width/8
# figure_width  = grid_width*6
# figure_heigth = grid_width*3
### small edition
grid_width    = colume_width/4
figure_width  = grid_width*4
figure_heigth = grid_width*2
# ----------------------------------------------------------------------------------------
fig,axes = plt.subplots(nrows=2,ncols=N_methods_show,figsize=(figure_width,figure_heigth),dpi=600,constrained_layout=True)
axs_map,axs_box = axes[0,:],axes[1,:]
[ax.set_axis_off() for ax in axes.ravel()]
# Parameter maps
enlarge = 20
for i,ax in enumerate(axs_map):
    pcm_map = ax.imshow(maps_pred[id_methods_show[i],id_repeat_show,id_slice_show,enlarge*2:,enlarge:-enlarge,1],cmap='jet',vmin=0.0,vmax=map_max)
    ax.set_title(method[id_methods_show[i]],font=font_dir,fontsize=font_size)
    rect = patches.Rectangle((x_box-enlarge,y_box-enlarge*2,),width=size_box,height=size_box,edgecolor='black',facecolor='none',linewidth=line_width)
    ax.add_patch(rect)
cbar = fig.colorbar(pcm_map,ax=axs_map,shrink=0.7,aspect=24.0,location='left',ticks=[0,5,10,15,20])
cbar.ax.set_yticklabels([0,5,10,15,20],font=font_dir,fontdict={'fontsize':font_size_tick})
# Box
for i,ax in enumerate(axs_box):
    pcm_box = ax.imshow(maps_pred[id_methods_show[i],id_repeat_show,id_slice_show,y_box:y_box+size_box,x_box:x_box+size_box,1],cmap='jet',vmin=0.0,vmax=map_max)
cbar = fig.colorbar(pcm_box,ax=axs_box,shrink=0.7,aspect=24.0,location='left',ticks=[0,5,10,15,20])
cbar.ax.set_yticklabels([0,5,10,15,20],font=font_dir,fontdict={'fontsize':font_size_tick})
plt.savefig(os.path.join('figures','maps_predict_invivo'))

##########################################################################################
# Linearity
r2_pix = np.squeeze(r2_pix) #(N_methods,N_pixel)
r2_ref = r2_pix[0]

# Linear regression
lin_results = []
for i in range(5):
    result = scipy.stats.linregress(r2_ref,r2_pix[i])
    lin_results.append(result)
    print('{:12}| R-Square: {:.4f}, Regression Slope: {:.4f}, Rregression intercept: {:.4f}, P value: {}'\
        .format(method[i],(result.rvalue)**2, result.slope, result.intercept, result.pvalue))
# ----------------------------------------------------------------------------------------
font_size      = 6
font_size_tick = 6
line_width     = 0.5
marker_size    = 0.6
marker_edge_width = 0.01
left_x, left_y = 2, 2
font_size_metrix = 6
# ----------------------------------------------------------------------------------------
### large edition
grid_width    = page_width/8
figure_width  = grid_width*5.5
figure_heigth = grid_width*1.6
### small edition
grid_width    = colume_width/3
figure_width  = grid_width*3
figure_heigth = grid_width
# ----------------------------------------------------------------------------------------
# fig,axes = plt.subplots(nrows=1,ncols=N_methods_show,figsize=(figure_width,figure_heigth),dpi=600,constrained_layout=True)
fig,axes = plt.subplots(nrows=1,ncols=N_methods_show-1,figsize=(figure_width,figure_heigth),dpi=600,constrained_layout=True)
# axes[0].set_axis_off()
# axs_lin = axes[1:]
axs_lin = axes
for i,ax in enumerate(axs_lin): 
    ax.plot(r2_ref,r2_pix[id_methods_show[i+1]],'o',label=method[id_methods_show[i+1]],\
        color='red',markersize=marker_size,markerfacecolor='none',markeredgewidth=marker_edge_width)
    ax.set_xlabel('R$_2$ ('+method[id_methods_show[0]]+') (s$^{-1}$)',font=font_dir,fontsize=font_size)
    ax.plot([0.,map_max],[0.,map_max],'--',color='black',linewidth=line_width)
    ax.set_xlim([0.0,map_max])
    ax.set_ylim([0.0,map_max])
    ax.set_xticks(ticks=[0,5,10,15,20])
    ax.set_xticklabels(labels=[0,5,10,15,20])
    ax.tick_params(axis='x',labelsize=font_size_tick,width=line_width,length=2.0)

    res = lin_results[id_methods_show[i+1]]
    # ax.text(x=left_x,y=left_y,s='$R^2$={:.3f}, P value={:.3f}'.format((res.rvalue)**2,res.pvalue),\
    #     color='black',font=font_dir,fontsize=font_size_metrix)
    # ax.plot([0.,map_max],[res.intercept,res.slope*map_max+res.intercept],'--',color='blue',linewidth=line_width)
    ax.set_ylabel('R$_{2}$ ('+method[id_methods_show[i+1]]+') (s$^{-1}$)',font=font_dir,fontsize=font_size)
    ax.set_aspect('equal')

# axs_lin[0].set_ylabel('R$_\mathrm{2}$ ($s^{\mathrm{-1}}$)',font=font_dir,fontsize=font_size)
axs_lin[0].tick_params(axis='y',labelsize=font_size_tick,width=line_width,length=2.0)
axs_lin[1].tick_params(axis='y',labelleft=False,width=line_width,length=2.0)
axs_lin[2].tick_params(axis='y',labelleft=False,width=line_width,length=2.0)

plt.savefig(os.path.join('figures','maps_predict_invivo_lin'))

##########################################################################################
print('='*98)