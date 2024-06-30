import matplotlib.pyplot as plt
import os
import data_processor as dp
import numpy as np

dir_model   = os.path.join('data','brainweb','anatomical_model') # BrainWeb database
dir_subject = os.listdir(dir_model) # all subjects in database

sub_show   = 'subject54'
slice_show = np.arange(120,260,20,dtype=np.int)

crisp,fuzzy=dp.read_models(dir_model,sub_show)
print(crisp.shape,fuzzy.shape)

# show model
ncol = 6
nrow = 3
fig,axes = plt.subplots(nrows=nrow,ncols=ncol,figsize=(ncol*4,nrow*4))
[ax.set_axis_off() for ax in axes.ravel()]
for i in range(ncol):
    axes[0,i].imshow(fuzzy[slice_show[0],...,i],cmap='hot')
    axes[1,i].imshow(fuzzy[slice_show[0],...,i+6],cmap='hot')
    axes[2,i].imshow(crisp[slice_show[i],...],cmap='hot')
    axes[2,i].set_title(slice_show[i])
plt.savefig('figures/models')


