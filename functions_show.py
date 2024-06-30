import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
import numpy as np
import functions as func
import scipy

def rescale_map(map,type_show,type_map,scale=1.0):
    map_scale = np.zeros_like(map)
    map_scale[...,0] = map[...,0]
    if type_show == 'T2':
        if type_map == 'T2': map_scale[...,1] = map[...,1]*scale
        if type_map == 'R2': map_scale[...,1] = tf.math.divide_no_nan(1.0,map[...,1]/scale)
    if type_show == 'R2':
        if type_map == 'T2': map_scale[...,1] = tf.math.divide_no_nan(1000.0, map[...,1]*scale)
        if type_map == 'R2': map_scale[...,1] = map[...,1]/scale*1000.0
    return map_scale

def evaluate_mo(maps,maps_gt,mask,show_info=True,ave = True):
    if len(maps.shape)==5:
        Ns,N,Ny,Nx,Np = maps.shape
        print(':: Num of stage = {}, Num of Slice = {}, Matrix Size = {}*{}, Num of parameters = {}'.format(Ns,N,Ny,Nx,Np))
        nrmse_m0,nrmse_p2,ssim_m0,ssim_p2 = [],[],[],[]
        for i in range(Ns):
            nrmse_m0.append(func.nRMSE(y_pred=maps[i,...,0][...,np.newaxis],y_true=maps_gt[...,0][...,np.newaxis],roi=mask,average=ave))
            nrmse_p2.append(func.nRMSE(y_pred=maps[i,...,1][...,np.newaxis],y_true=maps_gt[...,1][...,np.newaxis],roi=mask,average=ave))
            ssim_m0.append(func.SSIM(y_pred=maps[i,...,0][...,np.newaxis],y_true=maps_gt[...,0][...,np.newaxis],roi=mask,average=ave))
            ssim_p2.append(func.SSIM(y_pred=maps[i,...,1][...,np.newaxis],y_true=maps_gt[...,1][...,np.newaxis],roi=mask,average=ave))
        nrmse_m0 = np.array(nrmse_m0)
        nrmse_p2 = np.array(nrmse_p2)
        ssim_m0  = np.array(ssim_m0)
        ssim_p2  = np.array(ssim_p2)
    if len(maps.shape)==4:
        N,Ny,Nx,Np = maps.shape
        if show_info: print(':: Num of Slice = {}, Matrix Size = {}*{}, Num of parameters = {}'.format(N,Ny,Nx,Np))
        nrmse_m0 = func.nRMSE(y_pred=maps[...,0][...,np.newaxis],y_true=maps_gt[...,0][...,np.newaxis],roi=mask,average=ave)
        nrmse_p2 = func.nRMSE(y_pred=maps[...,1][...,np.newaxis],y_true=maps_gt[...,1][...,np.newaxis],roi=mask,average=ave)
        ssim_m0  = func.SSIM(y_pred=maps[...,0][...,np.newaxis],y_true=maps_gt[...,0][...,np.newaxis],roi=mask,average=ave)
        ssim_p2  = func.SSIM(y_pred=maps[...,1][...,np.newaxis],y_true=maps_gt[...,1][...,np.newaxis],roi=mask,average=ave)
    return  np.array([nrmse_m0,nrmse_p2,ssim_m0,ssim_p2])

def significance_test(a,b,type='wilcoxon',equal_var='False',return_statistic=True):
    if type == 'wilcoxon': s,p = scipy.stats.wilcoxon(a,b)
    if type == 'student' : s,p = scipy.stats.ttest_ind(a,b,equal_var=equal_var)
    if type == 'student_p' : s,p = scipy.stats.ttest_rel(a,b)
    if return_statistic == True:  return s,p
    if return_statistic == False: return p

def scatter_hist_plot(fig,grid,x,y,labels,title=None,xlim=(0.0,16.0),ylim=(0.0,16.0),bins=100,countlim=(0,10000),show_hist=True):
    '''
    fig   = plt.figure(figsize=(3*N_methods,3),dpi=300)
    outer = gridspec.GridSpec(1,N_methods, wspace=0.2, hspace=0.2)
    labels = ['CSF','White Matter','Gray Matter','Vessel']
    for i in range(N_methods):
        func_show.scatter_hist_plot(fig,outer[i],x,y(maps_pred[i]),title=method[i],labels=labels,show_hist=False)
    '''
    num_group = len(x)
    if show_hist == True:
        inner = gridspec.GridSpecFromSubplotSpec(4, 4, subplot_spec=grid, wspace=0.2, hspace=0.2)
        scatter_ax = fig.add_subplot(inner[1:,:-1])
        x_hist_ax  = fig.add_subplot(inner[0,:-1],xticklabels=[])
        y_hist_ax  = fig.add_subplot(inner[1:,-1],yticklabels=[])

        scatter_ax.set_xlim(xlim)
        scatter_ax.set_ylim(ylim)
        x_hist_ax.set_xlim(xlim)
        x_hist_ax.set_ylim(countlim)
        y_hist_ax.set_ylim(ylim)
        y_hist_ax.set_xlim(countlim)

        scatter_ax.plot(xlim,ylim,'--',color='black')

        for i in range(num_group):
            scatter_ax.plot(x[i],y[i],'o',markersize=0.5,alpha=0.2,label=labels[i])
            x_hist_ax.hist(x[i],bins, histtype='stepfilled',orientation='vertical')
            y_hist_ax.hist(y[i],bins, histtype='stepfilled',orientation='horizontal')
        
        scatter_ax.legend(markerscale=10.0,labelcolor='linecolor')

    if show_hist == False:
        inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=grid)
        scatter_ax = fig.add_subplot(inner[0,0])
        scatter_ax.set_xlim(xlim)
        scatter_ax.set_ylim(ylim)
        scatter_ax.set_ylabel('R2 ('+title+') ($s^{-1}$)')
        scatter_ax.set_xlabel('R2 (Reference) ($s^{-1}$)')
        scatter_ax.plot(xlim,ylim,'--',color='black')
        for i in range(num_group):
            scatter_ax.plot(x[i],y[i],'o',markersize=0.5,alpha=0.2,label=labels[i])
        scatter_ax.legend(markerscale=10.0)
