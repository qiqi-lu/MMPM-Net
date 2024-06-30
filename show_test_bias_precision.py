import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import numpy as np
import os
import pathlib

import functions as func
import functions_show as func_show

##########################################################################################
sigma_test, type_te_test   = [0.01, 0.02, 0.05, 0.1, 0.15], 1
num_slice, num_realization = 1, 10
# id_method_show = [0,2,3,4]
id_method_show = [0,1,2,3,4]

name_resnet   = 'resnet_4_moled_r2_mae_mono_6_mix'
name_rim      = 'rim_Ns_6_f_36_r2_mmse_mean_6_mix_l2_0.001'
name_dopamine = 'dopamine_Ns_10_path_1_SW_0_Norm_0_r2_maemow_6_mix_dxx'
name_admmnet  = 'admmnet_Ns_15_Nk_5_Nt_1_f_3_path_1_r2_maemow_6_mix_xfzz_tt_0.001_0.1'

name_methods  = ['MLE','ResNet','RIM','DOPAMINE','MMPM-Net']
color_methods = ['purple','black','green','blue','red']
xlabels = sigma_test

# ---------------------------------------------------------------------------------------
sigma_train, type_te_train, scale= 'mix', 6, 100.0
type_map_train_admmnet, type_map_train_dopamine, type_map_train_rim, type_map_train_resnet, type_map_mle = 'R2','R2','R2','R2','R2'
type_map_gt, type_map_show = 'R2', 'R2'

get_nonzero = lambda x,roi:x[np.nonzero(roi)]
num_noise_level = len(sigma_test)
# ---------------------------------------------------------------------------------------
colume_width = 3.5 # inches
page_width   = 7.16 # inches
font_dir     = pathlib.Path(os.path.join('fonts','Helvetica.ttf'))

##########################################################################################
print('='*98)
rbvn,rcvn = [],[]
print('Load results ...')
for i in range(num_noise_level):
    dir_results = os.path.join('results','NLtst_{}_NLtrn_{}_TEtst_{}_TEtrn_{}_S_{}_N_{}'\
                    .format(sigma_test[i],sigma_train,type_te_test,type_te_train,num_slice,num_realization))
    print('> ',dir_results)
    maps_mle             = np.load(os.path.join(dir_results,'maps_mle.npy'))
    maps_pred_resnet     = np.load(os.path.join(dir_results,name_resnet,'maps_pred_resnet.npy'))
    maps_pred_rim_m      = np.load(os.path.join(dir_results,name_rim,'maps_pred_rim.npy'))
    maps_pred_dopamine_m = np.load(os.path.join(dir_results,name_dopamine,'maps_pred_dopamine.npy'))
    maps_pred_admmnet_m  = np.load(os.path.join(dir_results,name_admmnet,'maps_pred_admmnet.npy'))

    maps_gt = np.load(os.path.join(dir_results,'maps_gt_test.npy'))
    seg     = np.load(os.path.join(dir_results,'seg_test.npy'))

    maps_mle_rescale = func_show.rescale_map(maps_mle,type_show=type_map_show,type_map=type_map_gt,scale=scale)
    maps_pred_resnet_recale = func_show.rescale_map(maps_pred_resnet,type_show=type_map_show,type_map=type_map_train_resnet,scale=scale)
    maps_pred_rim_m_recale  = func_show.rescale_map(maps_pred_rim_m,type_show=type_map_show,type_map=type_map_train_rim,scale=scale)
    maps_pred_dopamine_m_recale = func_show.rescale_map(maps_pred_dopamine_m,type_show=type_map_show,type_map=type_map_train_dopamine,scale=scale)
    maps_pred_admmnet_m_recale  = func_show.rescale_map(maps_pred_admmnet_m,type_show=type_map_show,type_map=type_map_train_admmnet,scale=scale)
    maps_gt_rescale  = func_show.rescale_map(maps_gt,type_show=type_map_show,type_map=type_map_gt,scale=scale)

    maps_pred = []
    maps_pred.append(maps_mle_rescale)
    maps_pred.append(maps_pred_resnet_recale)
    maps_pred.append(maps_pred_rim_m_recale[-1])
    maps_pred.append(maps_pred_dopamine_m_recale[-1])
    maps_pred.append(maps_pred_admmnet_m_recale[-1])

    ROI_wh = np.where(seg == 1,1,0)
    for i in [2,3,8]: ROI_wh  = ROI_wh + np.where(seg == i,1,0)
    ROI = func.closing_hole(ROI_wh)
    mask_bkg  = np.where(maps_gt_rescale > 0.0,1.0,0.0)

    # Bias and RC
    rbv, rcv = [], []
    roi_reshape = func.reshape_realization(ROI[...,0],Nslice=num_slice,Nrealization=num_realization)
    for i,maps in enumerate(maps_pred):
        maps   = maps*mask_bkg
        rb_map = func.relative_bias_map(Y=maps[...,1],X=maps_gt_rescale[...,1],Nslice=num_slice,Nrealization=num_realization)
        rc_map = func.repeatability_coefficient_map(Y=maps[...,1],Nslice=num_slice,Nrealization=num_realization)
        rbv.append(get_nonzero(rb_map,roi_reshape[0]))
        rcv.append(get_nonzero(rc_map,roi_reshape[0]))
    rbvn.append(rbv)
    rcvn.append(rcv)

rbvn = np.stack(rbvn)
rcvn = np.stack(rcvn)

##########################################################################################
font_size        = 8   # 10, 8
font_size_tick   = 6   # 8, 6
font_size_legend = 5
box_line_width   = 0.5 # 0.75, 0.5
# -----------------------------------------------------------------------------------------
boxprops = dict(linewidth=box_line_width)
capprops = dict(linewidth=box_line_width)
whiskerprops = dict(linewidth=box_line_width)
medianprops  = dict(linewidth=box_line_width)
# -----------------------------------------------------------------------------------------
pos = np.arange(num_noise_level)
# -----------------------------------------------------------------------------------------
### large edition
# grid_width    = page_width/8
### small edition
grid_width    = colume_width/6

figure_width  = grid_width*3
figure_heigth = grid_width*2.25
# -----------------------------------------------------------------------------------------
legend_patch, legend_name = [], []
for i in range(len(name_methods)):
    legend_patch.append(mpatch.Patch(edgecolor=color_methods[i],facecolor='white',linewidth=box_line_width))
    legend_name.append(name_methods[i])
# -----------------------------------------------------------------------------------------
# Bias
fig, axes = plt.subplots(nrows=1,ncols=1,figsize=(figure_width,figure_heigth),dpi=600,constrained_layout=True)
axes.axhline(y=0.0, lw=box_line_width, color='black')

for i,idx in enumerate(id_method_show):
    bx_rb = axes.boxplot(np.transpose(rbvn[:,idx],axes=(1,0)),positions=pos+0.15*i, widths=0.1,showfliers=False,\
                            boxprops=boxprops,whiskerprops=whiskerprops,capprops=capprops,medianprops=medianprops)
    plt.setp(bx_rb['boxes'], color=color_methods[idx])

limit_bias = [-15.0,15.0]
axes.set_xticks(pos+0.225)
axes.set_xticklabels(xlabels)
axes.set_yticks([-15,-10,-5,0,5,10,15])
axes.set_yticklabels([-15,-10,-5,0,5,10,15])
axes.tick_params(axis='y', labelsize=font_size_tick,width=box_line_width,length=2.0, direction='in')
axes.tick_params(axis='x', labelsize=font_size_tick,width=box_line_width,length=0.0)
plt.setp(axes.spines.values(), linewidth=box_line_width)
axes.grid(axis='y',linewidth=box_line_width*0.5)

axes.set_xlabel('Noise SD',font=font_dir,fontsize=font_size)
axes.set_ylabel('Bias (%)',font=font_dir,fontsize=font_size)
axes.set_ylim(limit_bias)

# axes.legend(legend_patch,legend_name,prop={'size': font_size_legend})
plt.savefig('figures/noise_vs_bias')

# -----------------------------------------------------------------------------------------
# RC
fig, axes = plt.subplots(nrows=1,ncols=1,figsize=(figure_width,figure_heigth),dpi=600,constrained_layout=True)
for i,idx in enumerate(id_method_show):
    bx_rc = axes.boxplot(np.transpose(rcvn[:,idx],axes=(1,0)),positions=pos+0.15*i, widths=0.1,showfliers=False,\
                            boxprops=boxprops,whiskerprops=whiskerprops,capprops=capprops,medianprops=medianprops)
    plt.setp(bx_rc['boxes'], color=color_methods[idx])

limit_rc = [0.0,60.0]
axes.set_xticks(pos+0.225)
axes.set_xticklabels(xlabels)
axes.set_yticks([0,10,20,30,40,50,60])
axes.set_yticklabels([0,10,20,30,40,50,60])
axes.grid(axis='y',linewidth=box_line_width*0.5)
axes.tick_params(axis='y', labelsize=font_size_tick, width=box_line_width, length=2.0, direction='in')
axes.tick_params(axis='x', labelsize=font_size_tick, width=box_line_width, length=0.0, direction='in')
plt.setp(axes.spines.values(), linewidth=box_line_width)


axes.set_xlabel('Noise SD',font=font_dir,fontsize=font_size)
axes.set_ylabel('RC (%)',font=font_dir,fontsize=font_size)
axes.set_ylim(limit_rc)

# axes.legend(legend_patch,legend_name,prop={'size': font_size_legend})
plt.savefig('figures/noise_vs_rc')

##########################################################################################
print('='*98)