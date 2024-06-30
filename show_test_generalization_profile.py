import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import numpy as np
import os
import pathlib

import functions as func
import functions_show as func_show

##########################################################################################
print('='*98)
sigma_test, type_te_test   = 0.05, [0,1,2,3,4,5,6,7,8,9,10,11,12,13]

name_resnet   = 'resnet_4_moled_r2_mae_mono_6_mix'
name_rim      = 'rim_Ns_6_f_36_r2_mmse_mean_6_mix_l2_0.001'
name_dopamine = 'dopamine_Ns_10_path_1_SW_0_Norm_0_r2_maemow_6_mix_dxx'
name_admmnet  = 'admmnet_Ns_15_Nk_5_Nt_1_f_3_path_1_r2_maemow_6_mix_xfzz_tt_0.001_0.1'

lables = ['MLE','ResNet','RIM','DOPAMINE','MMPM-Net']
colors = ['purple','black','green','blue','red']

id_method_show_ste = [0,1,2,3,4]
id_method_show_dte = [0,2,3,4]

id_protocol_show_ste = np.r_[0,1,4,5,9]
id_protocol_show_dte = np.r_[7,10,11,12,13]

# tick_lable_ste = id_protocol_show_ste
# tick_lable_dte = id_protocol_show_dte

tick_lable_ste = [1,2,3,4,5]
tick_lable_dte = [6,7,8,9,10]
# ----------------------------------------------------------------------------------------
sigma_train, type_te_train = 'mix', 6
num_slice, num_realization = 1, 10
type_map_train_admmnet, type_map_train_dopamine, type_map_train_rim, type_map_train_resnet, type_map_mle  = 'R2','R2','R2','R2','R2'
type_map_gt, type_map_show = 'R2','R2'
scale = 100.0

get_nonzero  = lambda x,roi:x[np.nonzero(roi)]
N_experiment = len(type_te_test)
# ----------------------------------------------------------------------------------------
colume_width = 3.5 # inches
page_width   = 7.16 #inches
font_dir     = pathlib.Path(os.path.join('fonts','Helvetica.ttf'))

##########################################################################################
nrmse_ssim, rb, rc = [], [], []
print('Load results ...')
for i in range(N_experiment):
    dir_results = os.path.join('results','NLtst_{}_NLtrn_{}_TEtst_{}_TEtrn_{}_S_{}_N_{}'\
                                .format(sigma_test,sigma_train,type_te_test[i],type_te_train,num_slice,num_realization))
    print('> ',dir_results)
    maps_mle             = np.load(os.path.join(dir_results,'maps_mle.npy'))
    maps_pred_resnet     = np.load(os.path.join(dir_results,name_resnet,'maps_pred_resnet.npy'))
    maps_pred_rim_m      = np.load(os.path.join(dir_results,name_rim,'maps_pred_rim.npy'))
    maps_pred_dopamine_m = np.load(os.path.join(dir_results,name_dopamine,'maps_pred_dopamine.npy'))
    maps_pred_admmnet_m  = np.load(os.path.join(dir_results,name_admmnet,'maps_pred_admmnet.npy'))
    maps_gt  = np.load(os.path.join(dir_results,'maps_gt_test.npy'))
    seg      = np.load(os.path.join(dir_results,'seg_test.npy'))

    maps_mle_rescale            = func_show.rescale_map(maps_mle,            type_show=type_map_show,type_map=type_map_mle,             scale=scale)
    maps_pred_resnet_recale     = func_show.rescale_map(maps_pred_resnet,    type_show=type_map_show,type_map=type_map_train_resnet,   scale=scale)
    maps_pred_rim_m_recale      = func_show.rescale_map(maps_pred_rim_m,     type_show=type_map_show,type_map=type_map_train_rim,      scale=scale)
    maps_pred_dopamine_m_recale = func_show.rescale_map(maps_pred_dopamine_m,type_show=type_map_show,type_map=type_map_train_dopamine, scale=scale)
    maps_pred_admmnet_m_recale  = func_show.rescale_map(maps_pred_admmnet_m, type_show=type_map_show,type_map=type_map_train_admmnet,  scale=scale)
    maps_gt_rescale             = func_show.rescale_map(maps_gt,             type_show=type_map_show,type_map=type_map_gt,             scale=scale)

    # Evaluation on last output maps and mask out background
    maps_pred = []
    maps_pred.append(maps_mle_rescale)
    maps_pred.append(maps_pred_resnet_recale)
    maps_pred.append(maps_pred_rim_m_recale[-1])
    maps_pred.append(maps_pred_dopamine_m_recale[-1])
    maps_pred.append(maps_pred_admmnet_m_recale[-1])

# ----------------------------------------------------------------------------------------
    ROI_wh = np.where(seg == 1,1,0)
    for i in [2,3,8]: ROI_wh = ROI_wh + np.where(seg == i,1,0)
    ROI_brain = func.closing_hole(ROI_wh)
    mask_bkg  = np.where(maps_gt_rescale > 0.0,1.0,0.0)

# ----------------------------------------------------------------------------------------
# NRMSE and SSIM
    m = []
    for i,maps in enumerate(maps_pred): 
        maps_pred[i] = maps*mask_bkg
        x = func_show.evaluate_mo(maps=maps_pred[i],maps_gt=maps_gt_rescale,mask=ROI_brain,show_info=False,ave=True)
        m.append(x)
    nrmse_ssim.append(m)
# ----------------------------------------------------------------------------------------
# Bias and RC
    rbv, rcv = [], []
    roi_reshape = func.reshape_realization(ROI_brain[...,0],Nslice=num_slice,Nrealization=num_realization)
    for i,maps in enumerate(maps_pred):
        maps   = maps*mask_bkg
        rb_map = func.relative_bias_map(Y=maps[...,1],X=maps_gt_rescale[...,1],Nslice=num_slice,Nrealization=num_realization)
        rc_map = func.repeatability_coefficient_map(Y=maps[...,1],Nslice=num_slice,Nrealization=num_realization)
        rbv.append(get_nonzero(rb_map,roi_reshape[0]))
        rcv.append(get_nonzero(rc_map,roi_reshape[0]))
    rb.append(rbv)
    rc.append(rcv)

rb = np.stack(rb) # [N_te,N_method,N_pixel]
rc = np.stack(rc) 
nrmse_ssim = np.stack(nrmse_ssim) # [N_te,N_method,[nrmse_m0,ssim_m0,nrmse_r2,ssim_r2]]

##########################################################################################
# NRMSE and SSIM profile
vlimit = [[0.0,0.35],[0.4,1.0]]
# ----------------------------------------------------------------------------------------
font_size        = 10 # 10, 8
font_size_tick   = 8  # 8, 6
font_size_legend = 6
line_width       = 1.0 # 0.75, 0.5
# ----------------------------------------------------------------------------------------
grid_width    = page_width/8
figure_width  = grid_width*6
figure_heigth = grid_width*4
# ----------------------------------------------------------------------------------------
fig,axes = plt.subplots(nrows=2,ncols=2,dpi=600,figsize=(figure_width,figure_heigth),constrained_layout=True)
for i in id_method_show_ste: 
    axes[0,0].plot(nrmse_ssim[id_protocol_show_ste,i,1],'.-',color=colors[i],label=lables[i],linewidth=line_width)
    axes[1,0].plot(nrmse_ssim[id_protocol_show_ste,i,3],'.-',color=colors[i],label=lables[i],linewidth=line_width)

for i in id_method_show_dte:
    axes[0,1].plot(nrmse_ssim[id_protocol_show_dte,i,1],'.-',color=colors[i],label=lables[i],linewidth=line_width)
    axes[1,1].plot(nrmse_ssim[id_protocol_show_dte,i,3],'.-',color=colors[i],label=lables[i],linewidth=line_width)

for j in range(2):
    axes[0,j].set_ylim(vlimit[0])
    axes[1,j].set_ylim(vlimit[1])
    axes[1,j].set_xlabel('TE',fontsize=font_size,font=font_dir)

axes[0,0].set_ylabel('NRMSE',fontsize=font_size,font=font_dir)
axes[1,0].set_ylabel('SSIM',fontsize=font_size,font=font_dir)
axes[0,1].legend(prop={'size':font_size_legend})
axes[1,0].legend(prop={'size':font_size_legend})

for j in range(2):
    axes[j,0].set_xticks(np.arange(id_protocol_show_ste.shape[0])), axes[j,0].set_xticklabels(tick_lable_ste)
    axes[j,1].set_xticks(np.arange(id_protocol_show_dte.shape[0])), axes[j,1].set_xticklabels(tick_lable_dte)

axes[0,0].set_title('Same Number of TE',font=font_dir,fontsize=font_size)
axes[0,1].set_title('Different Number of TE',font=font_dir,fontsize=font_size)

for ax in axes.ravel(): 
    ax.grid(axis='y',linewidth=line_width)
    ax.tick_params(axis='y', labelsize=font_size_tick,width=line_width,length=2.0)
    ax.tick_params(axis='x', labelsize=font_size_tick,width=line_width,length=2.0)
    plt.setp(ax.spines.values(), linewidth=line_width)
plt.savefig(os.path.join('figures','nrmse_ssim_vs_te'))

##########################################################################################
# Bias and RC profile
font_size        = 8 # 10, 8
font_size_tick   = 6  # 8, 6
font_size_legend = 5
box_line_width   = 0.5 # 0.75 0.5
line_width       = 0.5
# -----------------------------------------------------------------------------------------
boxprops     = dict(linewidth=box_line_width)
capprops     = dict(linewidth=box_line_width)
whiskerprops = dict(linewidth=box_line_width)
medianprops  = dict(linewidth=box_line_width)
# -----------------------------------------------------------------------------------------
### large edition
# grid_width    = page_width/8
# figure_width  = grid_width*3
# figure_heigth = grid_width*2
### small edition
grid_width    = colume_width/6
figure_width  = grid_width*3
figure_heigth = grid_width*2.25
# -----------------------------------------------------------------------------------------
legend_patch, legend_name = [], []
for i in range(len(lables)):
    legend_patch.append(mpatch.Patch(edgecolor=colors[i],facecolor='white',linewidth=box_line_width))
    legend_name.append(lables[i])
# -----------------------------------------------------------------------------------------
# Same number of TE
pos = np.arange(len(id_protocol_show_ste))
# -----------------------------------------------------------------------------------------
# Bias 
fig, axes = plt.subplots(nrows=1,ncols=1,figsize=(figure_width,figure_heigth),dpi=600,constrained_layout=True)
axes.axhline(y=0.0, lw=line_width, color='black')

for i,idx in enumerate(id_method_show_ste):
    bx_rb = axes.boxplot(np.transpose(rb[id_protocol_show_ste,idx],axes=(1,0)),positions=pos+0.15*i, widths=0.1,showfliers=False,\
                            boxprops=boxprops,whiskerprops=whiskerprops,capprops=capprops,medianprops=medianprops)
    plt.setp(bx_rb['boxes'], color=colors[idx])

limit_bias = [-15.0,15.0]
axes.set_xticks(pos+0.225)
axes.set_xticklabels(tick_lable_ste)
axes.set_yticks([-15,-10,-5,0,5,10,15])
axes.set_yticklabels([-15,-10,-5,0,5,10,15])
axes.tick_params(axis='y', labelsize=font_size_tick,width=line_width,length=2.0,direction='in')
axes.tick_params(axis='x', labelsize=font_size_tick,width=line_width,length=0.0)
plt.setp(axes.spines.values(), linewidth=line_width)
axes.grid(axis='y',linewidth=line_width)

axes.set_xlabel('Index of TE setting',font=font_dir,fontsize=font_size)
axes.set_ylabel('Bias (%)',font=font_dir,fontsize=font_size)
axes.set_ylim(limit_bias)

# axes.legend(legend_patch,legend_name,prop={'size': font_size_legend})
plt.savefig('figures/te_vs_bias_same')

# -----------------------------------------------------------------------------------------
# RC
fig, axes = plt.subplots(nrows=1,ncols=1,figsize=(figure_width,figure_heigth),dpi=600,constrained_layout=True)
for i,idx in enumerate(id_method_show_ste):
    bx_rc = axes.boxplot(np.transpose(rc[id_protocol_show_ste,idx],axes=(1,0)),positions=pos+0.15*i, widths=0.1,showfliers=False,\
                            boxprops=boxprops,whiskerprops=whiskerprops,capprops=capprops,medianprops=medianprops)
    plt.setp(bx_rc['boxes'], color=colors[idx])

limit_rc = [0.0,60.0]
axes.set_xticks(pos+0.225)
axes.set_xticklabels(tick_lable_ste)
axes.set_yticks([0,10,20,30,40,50,60])
axes.set_yticklabels([0,10,20,30,40,50,60])
axes.grid(axis='y',linewidth=line_width)
axes.tick_params(axis='y', labelsize=font_size_tick, width=line_width, length=2.0,direction='in')
axes.tick_params(axis='x', labelsize=font_size_tick, width=line_width, length=0.0)
plt.setp(axes.spines.values(), linewidth=line_width)

axes.set_xlabel('Index of TE setting',font=font_dir,fontsize=font_size)
axes.set_ylabel('RC (%)',font=font_dir,fontsize=font_size)
axes.set_ylim(limit_rc)

# axes.legend(legend_patch,legend_name,prop={'size': font_size_legend})
plt.savefig('figures/te_vs_rc_same')

# -----------------------------------------------------------------------------------------
# Different number of TE
pos = np.arange(len(id_protocol_show_dte))
# -----------------------------------------------------------------------------------------
# Bias 
fig, axes = plt.subplots(nrows=1,ncols=1,figsize=(figure_width,figure_heigth),dpi=600,constrained_layout=True)
axes.axhline(y=0.0, lw=line_width, color='black')

for i,idx in enumerate(id_method_show_dte):
    bx_rb = axes.boxplot(np.transpose(rb[id_protocol_show_dte,idx],axes=(1,0)),positions=pos+0.15*i, widths=0.1,showfliers=False,\
                            boxprops=boxprops,whiskerprops=whiskerprops,capprops=capprops,medianprops=medianprops)
    plt.setp(bx_rb['boxes'], color=colors[idx])

limit_bias = [-15.0,15.0]
axes.set_xticks(pos+0.225)
axes.set_xticklabels(tick_lable_dte)
axes.set_yticks([-15,-10,-5,0,5,10,15])
axes.set_yticklabels([-15,-10,-5,0,5,10,15])
axes.tick_params(axis='y', labelsize=font_size_tick,width=line_width,length=2.0,direction='in')
axes.tick_params(axis='x', labelsize=font_size_tick,width=line_width,length=0.0)
plt.setp(axes.spines.values(), linewidth=line_width)
axes.grid(axis='y',linewidth=line_width)

axes.set_xlabel('Index of TE setting',font=font_dir,fontsize=font_size)
axes.set_ylabel('Bias (%)',font=font_dir,fontsize=font_size)
axes.set_ylim(limit_bias)

# axes.legend(legend_patch,legend_name,prop={'size': font_size_legend})
plt.savefig('figures/te_vs_bias_diff')
# -----------------------------------------------------------------------------------------
# RC
fig, axes = plt.subplots(nrows=1,ncols=1,figsize=(figure_width,figure_heigth),dpi=600,constrained_layout=True)
for i,idx in enumerate(id_method_show_dte):
    bx_rc = axes.boxplot(np.transpose(rc[id_protocol_show_dte,idx],axes=(1,0)),positions=pos+0.15*i, widths=0.1,showfliers=False,\
                            boxprops=boxprops,whiskerprops=whiskerprops,capprops=capprops,medianprops=medianprops)
    plt.setp(bx_rc['boxes'], color=colors[idx])

limit_rc = [0.0,60.0]
axes.set_xticks(pos+0.225)
axes.set_xticklabels(tick_lable_dte)
axes.set_yticks([0,10,20,30,40,50,60])
axes.set_yticklabels([0,10,20,30,40,50,60])
axes.tick_params(axis='y', labelsize=font_size_tick, width=line_width, length=2.0,direction='in')
axes.tick_params(axis='x', labelsize=font_size_tick, width=line_width, length=0.0)
plt.setp(axes.spines.values(), linewidth=line_width)
axes.grid(axis='y',linewidth=line_width)

axes.set_xlabel('Index of TE setting',font=font_dir,fontsize=font_size)
axes.set_ylabel('RC (%)',font=font_dir,fontsize=font_size)
axes.set_ylim(limit_rc)

axes.legend(legend_patch,legend_name,prop={'size': font_size_legend})
plt.savefig('figures/te_vs_rc_diff')

##########################################################################################
print('='*98)