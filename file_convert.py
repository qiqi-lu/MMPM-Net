import numpy as np
import os
import glob
import functions as func

###### phantom mask raw to npy #######
dir_data    = os.path.join('data','invivodata')
name_study  = 'Study_30_1'
# name_folder = 'ROI'
name_folder = 'mask'

shape = (400,400)
# shape = (512,512)

file_dir   = os.path.join(dir_data,name_study,name_folder,'*.raw')
file_names = sorted(glob.glob(file_dir),key=lambda x :int(os.path.splitext(os.path.basename(x))[0]))

mask = []
for path in file_names:
    print(path)
    m = func.read_binary(file_name=path,data_type='int8',shape=shape)
    mask.append(m)
mask = np.stack(mask)
print(mask.shape)
np.save(file=os.path.join(dir_data,name_study,name_folder),arr=mask)