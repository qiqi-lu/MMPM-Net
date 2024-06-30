import matplotlib.pyplot as plt
import os
import data_processor as dp
import numpy as np

type_task = 'T1mapping'
# type_task = 'T2mapping'

dir_model   = os.path.join('data','brainweb','anatomical_model') # BrainWeb database
dir_subject = os.listdir(dir_model) # all subjects in database

sub_show   = 'subject54'
slice_show = 183
# slice_show = np.arange(120,260,20,dtype=np.int)

crisp,fuzzy=dp.read_models(dir_model,sub_show,ac=8)
print(crisp.shape,fuzzy.shape)

# show model
page_width = 7.16
fig,axes = plt.subplots(nrows=3,ncols=3,figsize=(page_width,page_width),tight_layout=True,dpi=600)
axes=axes.ravel()
for i,ax in enumerate(axes):
    ax.set_axis_off()
    ax.imshow(crisp[slice_show+i,...],cmap='hot')
    ax.set_title(slice_show+i)
plt.savefig(os.path.join('figures',type_task,'models'))


