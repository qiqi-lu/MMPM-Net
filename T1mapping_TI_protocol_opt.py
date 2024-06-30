# Optimization of TI settings in R1 mapping task.
# R1 = 1/T1
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as sio
import functions as func
import data_processor as dp

#######################################################################
page_width   = 7.16
colume_width = 3.5

#######################################################################
# T1 examples in normal human brain.
T1_example = np.array([200.0,400.0,800.0,1600.0,3200.0,4000.0]) # ms
R1_example = 1000.0/T1_example # s

# Parameter settings.
TIs_example = np.array([50.0,100.0,200.0,400.0,800.0,1600.0,3200.0,6400.0]) # ms
alpha = 4 # degree
tau = 6 # ms

# T1 to T1*.
T1s_example = func.T1_to_T1s(T1_example,FA=alpha,TR=tau)

print('T1:     ',T1_example)
print('T1*:    ',T1s_example)
print('T1*/T1: ',T1s_example/T1_example)

#######################################################################
# signal generation
def ThreePara(A,B,T1,TIs,signal_type='signed'):
    'Three parameter model: s = A-B*exp(-TIs/T1)'
    if signal_type == 'signed':
        s = A-B*np.exp(-TIs/T1)
    if signal_type == 'unsigned':
        s = np.abs(A-B*np.exp(-TIs/T1))
    return s

tis = np.array([50.0,100.0,200.0,400.0,800.0,1600.0,3200.0,6400.0])
M0 = 1.0
M0s = (M0*T1s_example/T1_example)[...,np.newaxis]
A = M0s
B = M0s+M0

signals_LL = ThreePara(A=A,B=B,T1=T1s_example[...,np.newaxis],TIs=tis,signal_type='signed') # signal of Look-Locker
signals_IR = ThreePara(A=M0,B=2*M0,T1=T1s_example[...,np.newaxis],TIs=tis,signal_type='signed') # signal with common IR sequence 

fig,axes = plt.subplots(nrows=1,ncols=1,dpi=600,figsize=(colume_width,colume_width),tight_layout=True)
colors = ['rosybrown','lightcoral','indianred','brown','firebrick','maroon']
axes.axhline(y=1.0,linestyle='--',color='black')
axes.axhline(y=0.0,linestyle='--',color='black')
axes.axhline(y=-1.0,linestyle='--',color='black')
for i in range(T1_example.shape[0]):
    axes.plot(tis,signals_IR[i],'--',marker='o',markersize=3,color=colors[i],label=str(T1_example[i])+' ms')
    axes.plot(tis,signals_LL[i],marker='o',markersize=3,color=colors[i])
axes.legend()
axes.set_xlabel('TI (ms)')
axes.set_ylabel('S')
plt.savefig(os.path.join('figures','T1mapping','signal_example'))


#######################################################################
# MAP datasets
folder = os.path.join('data','T1mapping','invivo_MAP')
A    = sio.loadmat(os.path.join(folder,'A.mat'))['A']
B    = sio.loadmat(os.path.join(folder,'B.mat'))['B']
T1s  = sio.loadmat(os.path.join(folder,'T1s.mat'))['T1s']
TIs  = sio.loadmat(os.path.join(folder,'TIs.mat'))['TIs']
data = sio.loadmat(os.path.join(folder,'image.mat'))['data']

# T1 correction
T1_map = ((B/A)-1)*T1s
M0_map = B-A
BdA_map = B/A

# Simulation parameter setting
alpha = 4 # degree
tau   = 500 # ms

# Simulation
T1s_map_simu = func.T1_to_T1s(T1_map,FA=alpha,TR=tau)
A_map_simu   = M0_map*T1s_map_simu/T1_map
B_map_simu   = M0_map+A_map_simu

# Show results
fig, axes = plt.subplots(nrows=3,ncols=3,dpi=600,figsize=(page_width,page_width),tight_layout=True)
[ax.set_axis_off() for ax in axes.ravel()]
cb00 = axes[0,0].imshow(A,cmap='gray',vmax=1.0,vmin=0.0,interpolation='none')
axes[0,0].set_title('A (M0*)')
cb01 = axes[0,1].imshow(B,cmap='gray',vmax=2.0,vmin=0.0,interpolation='none')
axes[0,1].set_title('B (M0*+M0)')
cb02 = axes[0,2].imshow(T1s,cmap='jet',vmax=5000.0,vmin=0.0,interpolation='none')
axes[0,2].set_title('T1*')

cb10 = axes[1,0].imshow(M0_map,cmap='gray',vmax=1.0,vmin=0.0,interpolation='none')
axes[1,0].set_title('M0 (B-A)')
cb11 = axes[1,1].imshow(BdA_map,cmap='gray',vmax=3.0,vmin=0.0,interpolation='none')
axes[1,1].set_title('B/A')
cb12 = axes[1,2].imshow(T1_map,cmap='jet',vmax=5000.0,vmin=0.0,interpolation='none')
axes[1,2].set_title('T1 ((B/A-1)*T1*)')

cb20 = axes[2,0].imshow(A_map_simu,cmap='gray',vmax=1.0,vmin=0.0,interpolation='none')
axes[2,0].set_title('A_simu (M0*)')
cb21 = axes[2,1].imshow(B_map_simu,cmap='gray',vmax=2.0,vmin=0.0,interpolation='none')
axes[2,1].set_title('B_simu (M0*+M0)')
cb22 = axes[2,2].imshow(T1s_map_simu,cmap='jet',vmax=5000.0,vmin=0.0,interpolation='none')
axes[2,2].set_title('T1*_simu (alpha='+str(alpha)+')')

cbs = [cb00,cb01,cb02,cb10,cb11,cb12,cb20,cb21,cb22]
for c,ax in zip(cbs,axes.ravel()):
    fig.colorbar(c,ax=ax,shrink=0.65,aspect=20)

plt.savefig(os.path.join('figures','T1mapping','MAP_maps'))

#######################################################################
# The effect of flip angle (alpha) and TR (tau) for T1*/T1 (at specific T1)
alpha_range = [2,20]
tau_range   = [5,50]
grid_size  = 100
alpha_grid = np.linspace(start=alpha_range[0],stop=alpha_range[1],num=grid_size)[np.newaxis,...]
tau_grid   = np.linspace(start=tau_range[0],stop=tau_range[1],num=grid_size)[np.newaxis,...]
alpha_grid = np.repeat(alpha_grid,repeats=grid_size,axis=0)
tau_grid   = np.transpose(np.repeat(tau_grid,repeats=grid_size,axis=0))

# Calculated fraction maps
T1 = [250.0,500.0,1000.0,2000.0,4000.0]
fraction_maps = np.zeros(shape=(len(T1),grid_size,grid_size))
for i in range(len(T1)):
    fraction_maps[i] = func.T1_to_T1s(T1[i],FA=alpha_grid,TR=tau_grid)/T1[i]

# Show fraction maps
fig,axes = plt.subplots(nrows=2,ncols=3,figsize=(page_width,page_width/len(T1)*4),dpi=600,tight_layout=True)
axes = axes.ravel()
axes[-1].set_axis_off()
for i,map in enumerate(fraction_maps):
    # ax.set_axis_off()
    axes[i].imshow(map,vmax=1.0,vmin=0.0,cmap='gray',interpolation='none'),axes[i].set_title('T1='+str(T1[i]))
    axes[i].set_xticks([0,99])
    axes[i].set_xticklabels([alpha_range[0],alpha_range[1]])
    axes[i].set_yticks([0,99])
    axes[i].set_yticklabels([tau_range[0],tau_range[1]])
    axes[i].set_xlabel('FA (degree)')
    axes[i].set_ylabel('TR (ms)')
plt.savefig(os.path.join('figures','T1mapping','MAP_fraction_map'))


