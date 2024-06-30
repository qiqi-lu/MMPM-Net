import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib

import functions_show as func_show

##########################################################################################
name_protocol, type_te_test, show_resnet, show_legend = 'SE',   1,  1, 1
# name_protocol, type_te_test, show_resnet, show_legend = 'SE',   3,  1, 0
# name_protocol, type_te_test, show_resnet, show_legend = 'MESE', 13, 0, 0


te1_idx = 1
te1     = 10.0
# ----------------------------------------------------------------------------------------
name_data  = 'phantom'
name_study = ['Study_15_1','Study_15_2','Study_15_3']
B0         = 1.5

name_admmnet  = 'admmnet_Ns_15_Nk_5_Nt_1_f_3_path_1_r2_maemow_6_mix_xfzz_tt_0.001_0.1'
# name_admmnet  = 'admmnet_Ns_10_Nk_5_Nt_1_f_3_path_1_r2_maemow_6_mix_xfzz_tt_0.001_0.1_dncnn_16_lr3'
name_dopamine = 'dopamine_Ns_10_path_1_SW_0_Norm_0_r2_maemow_6_mix_dxx'
name_rim      = 'rim_Ns_6_f_36_r2_mmse_mean_6_mix_l2_0.001'
name_resnet   = 'resnet_4_moled_r2_mae_mono_6_mix'

method = ['MLE','MMPM-Net','DOPAMINE','RIM','ResNet']
colors = ['purple','red','blue','green','black']
id_methods_show = [0,4,3,2,1]

flag_mask_out = 1

if B0 == 3.0: t2_phantom = np.array([581.3, 403.5, 278.1, 190.94, 133.27, 96.89, 64.07, 46.42, 31.97, 22.56, 15.813, 11.237, 7.911, 5.592]) # @3T
if B0 == 1.5: t2_phantom = np.array([939.4, 594.3, 416.5, 267.0,  184.9,  140.6, 91.76, 64.84, 45.28, 30.62, 19.76,  15.99,  10.47, 8.15])  # @1.5T

id_roi_show     = np.r_[1:14]
id_slice_show   = 0
id_repeat_show  = 0
# ----------------------------------------------------------------------------------------
colume_width = 3.5
page_width   = 7.16
font_dir     = pathlib.Path(os.path.join('fonts','Helvetica.ttf'))
# ----------------------------------------------------------------------------------------
sigma_train, type_te_train, scale = 'mix', 6, 100.0
type_map_train_admmnet, type_map_train_dopamine, type_map_train_rim, type_map_train_resnet, type_map_mle = 'R2','R2','R2','R2','R2'
type_map_show = 'R2'

##########################################################################################
print('='*98)
print('Load results ...')
imgs_n, maps_pred_admmnet_m, maps_pred_dopamine_m, maps_pred_rim_m, maps_pred_resnet, maps_mle = [],[],[],[],[],[]
for name in name_study:
    type_application = '{}_{}_{}_NLtrn_{}_TEtst_{}_TEtrn_{}'.format(name_data,name,name_protocol,sigma_train,type_te_test,type_te_train)
    dir_results      = os.path.join('results',type_application)
    print('> ',dir_results)
    imgs_n.append(              np.load(os.path.join(dir_results,'imgs_n_test.npy')))
    maps_pred_admmnet_m.append( np.load(os.path.join(dir_results,name_admmnet,'maps_pred_admmnet.npy')))
    maps_pred_dopamine_m.append(np.load(os.path.join(dir_results,name_dopamine,'maps_pred_dopamine.npy')))
    maps_pred_rim_m.append(     np.load(os.path.join(dir_results,name_rim,'maps_pred_rim.npy')))
    maps_pred_resnet.append(    np.load(os.path.join(dir_results,name_resnet,'maps_pred_resnet.npy')))
    maps_mle.append(            np.load(os.path.join(dir_results,'maps_mle.npy')))

imgs_n = np.stack(imgs_n)
maps_pred_admmnet_m  = np.stack(maps_pred_admmnet_m)
maps_pred_dopamine_m = np.stack(maps_pred_dopamine_m)
maps_pred_rim_m  = np.stack(maps_pred_rim_m)
maps_pred_resnet = np.stack(maps_pred_resnet)
maps_mle = np.stack(maps_mle)

# ----------------------------------------------------------------------------------------
# Rescale maps
maps_pred_admmnet_m_recale  = func_show.rescale_map(maps_pred_admmnet_m, type_show=type_map_show,type_map=type_map_train_admmnet,  scale=scale)
maps_pred_dopamine_m_recale = func_show.rescale_map(maps_pred_dopamine_m,type_show=type_map_show,type_map=type_map_train_dopamine, scale=scale)
maps_pred_rim_m_recale      = func_show.rescale_map(maps_pred_rim_m,     type_show=type_map_show,type_map=type_map_train_rim,      scale=scale)
maps_pred_resnet_recale     = func_show.rescale_map(maps_pred_resnet,    type_show=type_map_show,type_map=type_map_train_resnet,   scale=scale)
maps_mle_rescale            = func_show.rescale_map(maps_mle,            type_show=type_map_show,type_map=type_map_mle,            scale=scale)

print('Image shape: ',imgs_n.shape)
print('Map shape (ADMM-REDNet): ',maps_pred_admmnet_m.shape)

##########################################################################################
# Evaluation on last output maps
print('='*98)
maps_pred = []
maps_pred.append(maps_mle_rescale)
maps_pred.append(maps_pred_admmnet_m_recale[:,-1])
maps_pred.append(maps_pred_dopamine_m_recale[:,-1])
maps_pred.append(maps_pred_rim_m_recale[:,-1])
maps_pred.append(maps_pred_resnet_recale)

maps_pred = np.stack(maps_pred) # (N_methods,N_repeat,N_slices,Ny,Nx,Nz)
print(maps_pred.shape)

N_methods,N_repeat,N_slices = maps_pred.shape[0:3]

# Mask out background
if flag_mask_out == 1:
    for i,name in enumerate(name_study):
        mask_bkg = np.load(file=os.path.join('data','phantomdata',name,'mask.npy'))
        mask_bkg = np.repeat(np.abs(mask_bkg)[...,np.newaxis],repeats=2,axis=-1)
        for j in range(N_methods): maps_pred[j,i] = maps_pred[j,i]*mask_bkg

##########################################################################################
print('='*98)
print('ROI Analsis ...')
r2_ref = 1000.0/t2_phantom
r2_pix = []
for i_study, name in enumerate(name_study):
    mask_phantom = np.load(file=os.path.join('data','phantomdata',name,'ROI.npy'))
    mask_phantom = np.abs(mask_phantom)
    N_mask = mask_phantom.shape[0]

    m = []
    for i_method in range(N_methods):
        a = []
        for i_slice in range(N_slices):
            b = []
            for i_mask in range(N_mask):
                pix_idx = np.where(mask_phantom[i_mask])
                pix     = np.nan_to_num(maps_pred[i_method,i_study,i_slice,...,1][pix_idx],nan=0.0,posinf=0.0,neginf=0.0)
                b.append(pix)
            a.append(b)
        m.append(a)
    r2_pix.append(m)

r2_pix = np.transpose(np.array(r2_pix),axes=(1,0,2,3,4))
print(r2_pix.shape)

# ----------------------------------------------------------------------------------------
r2_pix = r2_pix[...,id_roi_show,:]
r2_ref = r2_ref[id_roi_show]

N_methods_show = len(id_methods_show)
N_roi_show     = len(id_roi_show)

##########################################################################################
# large edition
map_max = 40.0

font_size      = 8.0
font_size_tick = 6.0

grid_width    = page_width/8
figure_width  = grid_width*6.0
figure_heigth = grid_width*1.6
# ----------------------------------------------------------------------------------------
fig,axes = plt.subplots(nrows=1,ncols=N_methods_show+1,figsize=(figure_width,figure_heigth),dpi=600,constrained_layout=True)
# Wieghted image
axes[0].imshow(imgs_n[id_repeat_show,id_slice_show,:,:,te1_idx],cmap='gray',vmin=0.0,vmax=1.2)
axes[0].set_axis_off()
axes[0].set_title('T$_\mathrm{2}$w',font=font_dir,fontsize=font_size)
axes[0].text(x=15,y=50,s='TE={} ms'.format(te1),font=font_dir,fontsize=font_size_tick,color='white')
# Predict maps
for i,idx in enumerate(id_methods_show):
    pcm_map = axes[i+1].imshow(maps_pred[idx,id_repeat_show,id_slice_show,:,:,1],cmap='jet',vmin=0.0,vmax=map_max)
    axes[i+1].set_title(method[idx],font=font_dir,fontsize=font_size)
    axes[i+1].set_axis_off()

if show_resnet == False: 
    axes[2].imshow(np.zeros_like(maps_pred[0,id_repeat_show,id_slice_show,:,:,1]),cmap='jet',vmin=0.0,vmax=map_max)

cbar  = fig.colorbar(pcm_map,ax=axes,shrink=0.6,aspect=24.0,orientation='horizontal',ticks=[0.0,10.0,20.0,30.0,40.0])
cbar.ax.set_xticklabels([0.0,10.0,20.0,30.0,40.0],font=font_dir,fontdict={'fontsize':font_size_tick})
plt.savefig(os.path.join('figures','maps_pred_phantom'))

##########################################################################################
# small edition
map_max        = 40.0
font_size      = 6
font_size_tick = 6
# ----------------------------------------------------------------------------------------
grid_width    = colume_width/5
figure_width  = grid_width*5
figure_heigth = grid_width*2.8
# ----------------------------------------------------------------------------------------
fig,axes = plt.subplots(nrows=2,ncols=N_methods_show,figsize=(figure_width,figure_heigth),dpi=600,constrained_layout=True)
# Wieghted image
[ax.set_axis_off() for ax in axes.ravel()]
enlarge = 30
axes[0,2].imshow(imgs_n[id_repeat_show,id_slice_show,enlarge:-enlarge,enlarge:-enlarge,te1_idx],cmap='gray',vmin=0.0,vmax=1.2)
axes[0,2].set_axis_off()
axes[0,2].set_title('T$_\mathrm{2}$w',font=font_dir,fontsize=font_size)
# axes[0,2].text(x=15,y=50,s='TE={} ms'.format(te1),font=font_dir,fontsize=font_size_tick,color='white')
# Predict maps
for i,idx in enumerate(id_methods_show):
    pcm_map = axes[1,i].imshow(maps_pred[idx,id_repeat_show,id_slice_show,enlarge:-enlarge,enlarge:-enlarge,1],cmap='jet',vmin=0.0,vmax=map_max)
    axes[1,i].set_title(method[idx],font=font_dir,fontsize=font_size)

if show_resnet == False: 
    axes[1,1].imshow(np.zeros_like(maps_pred[0,id_repeat_show,id_slice_show,:,:,1]),cmap='jet',vmin=0.0,vmax=map_max)

cbar  = fig.colorbar(pcm_map,ax=axes,shrink=0.6,aspect=24.0,orientation='horizontal',ticks=[0.0,10.0,20.0,30.0,40.0])
cbar.ax.set_xticklabels([0,10,20,30,40],font=font_dir,fontdict={'fontsize':font_size_tick})
plt.savefig(os.path.join('figures','maps_pred_phantom_small_'+name_protocol+'_'+str(type_te_test)))

##########################################################################################
font_size        = 10  # 10, 8
font_size_legend = 7   # 5, 7
font_size_tick   = 8   # 5, 6
line_width       = 1.0 # 0.5, 1.0
marker_size      = 4.0 # 2.0, 4.0
# ----------------------------------------------------------------------------------------
# Linearity
### small edition
# grid_width    = page_width/8
# figure_heigth = grid_width*1.8
# figure_width  = grid_width*1.8
### large edition
grid_width    = page_width/3
figure_heigth = grid_width
figure_width  = grid_width
# ----------------------------------------------------------------------------------------
fig, axes     = plt.subplots(nrows=1,ncols=1,figsize=(figure_width,figure_heigth),dpi=600,constrained_layout=True)
r2_pix_mean   = np.mean(r2_pix,axis=(1,2,-1)) # [N_methods, N_roi]

for i in id_methods_show: 
    if i == 4 and show_resnet == False: continue
    axes.plot(r2_ref,r2_pix_mean[i],'o',label=method[i],color=colors[i],markersize=marker_size,linewidth=line_width)
axes.plot([0.,max(r2_ref)+10.],[0.,max(r2_ref)+10.],'--',color='black',linewidth=line_width)

axes.set_ylabel('Estimated R$_\mathrm{2}$ (s$^{\mathrm{-1}}$)',font=font_dir,fontsize=font_size)
axes.set_xticks([0.,25.,50.,75.,100.,125.])
axes.set_xticklabels([0,25,50,75,100,125])
axes.set_yticks([0.,25.,50.,75.,100.,125.])
axes.set_yticklabels([0,25,50,75,100,125])
axes.tick_params(axis='y',labelsize=font_size_tick,width=line_width,length=2.0)
# axes.tick_params(axis='x',labelsize=font_size_tick,width=line_width,length=2.0,rotation=45.0)
axes.tick_params(axis='x',labelsize=font_size_tick,width=line_width,length=2.0)

xlabel = 'Reference R$_2$ (s$^{-1}$)'
axes.set_xlabel(xlabel,font=font_dir,fontsize=font_size)
axes.set_aspect('equal')
plt.setp(axes.spines.values(), linewidth=line_width)
axes.legend(prop={'size':font_size_legend})
plt.savefig(os.path.join('figures','roi_analysis_phantom_lin_'+name_protocol+'_'+str(type_te_test)))
# ----------------------------------------------------------------------------------------
r2_pix_std    = np.std(np.mean(r2_pix,axis=-1),axis=(1,2))
rb = (r2_pix_mean-r2_ref[np.newaxis])/r2_ref[np.newaxis]*100
rc = 2.77*r2_pix_std/r2_pix_mean*100
print('Methods:',method)
print('Average absolute relative bias: ', np.mean(np.abs(rb),axis=-1))
print('Average absolute RC           : ', np.mean(np.abs(rc),axis=-1))
# ----------------------------------------------------------------------------------------
# Relative bias and Repeatability Coefficent
figure_width  = grid_width*4.8
figure_heigth = grid_width*1.8
# ----------------------------------------------------------------------------------------
fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(figure_width,figure_heigth),dpi=600,constrained_layout=True)
axes[0].axhline(0, color='black', lw=line_width)
pos = np.arange(N_roi_show)
for i in id_methods_show:
    if i == 4 and show_resnet == False: continue
    axes[0].plot(pos,rb[i],'--o',color=colors[i],label=method[i],markersize=marker_size,linewidth=line_width)
    axes[1].plot(pos,rc[i],'--o',color=colors[i],label=method[i],markersize=marker_size,linewidth=line_width)

axes[0].set_ylabel('Bias (%)',fontsize=font_size)
axes[0].set_ylim([-70.,50.])

axes[1].set_ylabel('RC (%)',fontsize=font_size)
axes[1].set_ylim([0.,20.])

for ax in axes.ravel(): 
    ax.set_xticks(pos)
    ax.set_xticklabels(np.round(r2_ref,decimals=1))
    ax.tick_params(axis='y',labelsize=font_size_tick,width=line_width,length=2.0)
    ax.tick_params(axis='x',labelsize=font_size_tick,width=line_width,length=2.0,rotation=45.0)
    ax.grid(axis='y',linewidth=line_width*0.75)

xlabel = 'Reference R$_2$ (s$^{-1}$)'
for ax in axes.ravel():
    ax.set_xlabel(xlabel,font=font_dir,fontsize=font_size)
    plt.setp(ax.spines.values(), linewidth=line_width)

if show_legend: axes[1].legend(prop={'size':font_size_legend})
plt.savefig(os.path.join('figures','roi_analysis_phantom_bias&RC'))

##########################################################################################
print('='*98)