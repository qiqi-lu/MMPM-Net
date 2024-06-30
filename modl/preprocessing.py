import numpy as np
import glob
import SimpleITK as sitk
import os
import sys

def resample_sitk_image_series(out_size,images_sitk):
    """Resample SimpleITK image to specific size
    """
    input_size = images_sitk.GetSize()
    input_spacing = images_sitk.GetSpacing()
    
    output_size = (out_size[0],out_size[1],input_size[2])
    
    output_spacing = np.array([0.,0.,0.]).astype('float64')
    output_spacing[0] = input_size[0]*input_spacing[0]/output_size[0]
    output_spacing[1] = input_size[1]*input_spacing[1]/output_size[1]
    output_spacing[2] = input_size[2]*input_spacing[2]/output_size[2]
    
    transform = sitk.Transform()
    transform.SetIdentity()
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputOrigin(images_sitk.GetOrigin())
    resampler.SetOutputSpacing(output_spacing)
    resampler.SetOutputDirection(images_sitk.GetDirection())
    resampler.SetSize(output_size)
    images_sitk_resample = resampler.Execute(images_sitk)

    return images_sitk_resample

def clearup_clinical_data(file_dir):
    """
    Clean up all raw clinical data and reshape to the same size
    """
    print(':: Clear up clinical data...')
    filepath = os.path.join(file_dir, 'Study2*')
    # get all study name
    name_folders = sorted(glob.glob(filepath),key=os.path.getmtime,reverse=True)
    num_study  = np.array(name_folders).shape[0]
    data_study = np.zeros([num_study,64,128,12])

    # read data and info, and reshape to same size
    for id_study in range(num_study):
        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(name_folders[id_study])
         
        if not series_ids:
            print("ERROR: given directory dose not a DICOM series.")
            sys.exit(1)
         
        series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(name_folders[id_study],series_ids[0])
        series_reader = sitk.ImageSeriesReader()
        series_reader.SetFileNames(series_file_names)
        images = series_reader.Execute()
        # resahpe image into same size
        images_resample = resample_sitk_image_series([data_study.shape[-2],data_study.shape[-3]], images)
        images_array = sitk.GetArrayFromImage(images_resample)
        for id_image in range(data_study.shape[-1]):
            data_study[id_study,:,:,id_image] = images_array[id_image,:,:]   
    
    np.save(os.path.join('data_clinical', 'clinical_data'),data_study)
    return data_study

def get_clinical_data():
    """
    Get clinical daya with same size
    """
    data = np.load(os.path.join('..','data_clinical','clinical_data.npy'))
    print(':: Get the clinical data with same size...'+ '('+str(data.shape)+')')
    return data

def data_aug(img, mode=0):
    # aug data size
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))

def patching(data_study):
    """
    Augment the data (num_study,x,y,num_channel) into patches.
    """  
    import cv2

    aug_times = 1
    scales = [1, 0.9, 0.8, 0.7]
    patch_size = 32
    stride = 8
    num_study,h,w,c=data_study.shape
    data_patches = []
    
    print(':: Patching...')

    for id_study in range(num_study):
        for s in scales:
            h_scaled, w_scaled = int(h*s),int(w*s)
            img_scaled   = cv2.resize(data_study[id_study,:,:,:], (w_scaled,h_scaled), interpolation=cv2.INTER_CUBIC)
            # extract patches
            for i in range(0, h_scaled-patch_size+1, stride):
                for j in range(0, w_scaled-patch_size+1, stride):
                    x = img_scaled[i:i+patch_size, j:j+patch_size,:]
                    # data augmentation
                    for k in range(0, aug_times):
                        mode=np.random.randint(0,8)
                        x_aug = data_aug(x, mode=mode)
                        data_patches.append(x_aug)

    np.save(os.path.join('data_train_clinical', '%s%d'%('clinical_data_train_patches_')),data_patches)
    return data_patches
    
def get_clinical_data_patches(sigma_g):
    print('Get clinical data patches...')
    data = np.load(os.path.join('data_train_clinical','%s%d%s'%('clinical_data_train_patches_',sigma_g,'.npy')),allow_pickle=True).item()  
    return data



if __name__ == '__main__':
    data = get_clinical_data()

