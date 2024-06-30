import numpy as np
import os
import glob
import functions as func
import tqdm
import pandas as pd
import multiprocessing
import operator

###################################################################################################
print('Load model information ...')
tissue_para = pd.read_csv(os.path.join('data','brainweb','tissue_parameter_table.txt'),sep='\t')
data_type   = pd.read_csv(os.path.join('data','brainweb','data_type_table.txt'),sep='\t')
sub_info    = pd.read_csv(os.path.join('data','brainweb','subject_information_table.txt'),sep='\t')

###################################################################################################
def read_models(dir_model,sub_name,ac=8):
    """
    Read discrete model and fuzzy models.
    #### ARGUMENTS
    - database_dir, path of the database.
    - sub, subject name.

    #### RETURN
    - crisp_model, dicrete mdoel.   [Nz,Ny,Nx]
    - fuzzy models, fuzzy model.    [Nz,Ny,Nx,Nm]
    """
    print('Read models ...')
    models_dir = glob.glob(os.path.join(os.path.join(dir_model,sub_name),'*.rawb.gz')) # get all model files directory
    sub = sub_info[sub_info['subject_name']==sub_name]
    Nz  = sub['Nz'].values[0]
    Ny  = sub['Ny'].values[0]
    Nx  = sub['Nx'].values[0]
    type = sub['type'].values[0]
    range_max = sub['max'].values[0]
    range_min = sub['min'].values[0]
    data_format = sub['format'].values[0]

    # Discrete model
    crisp_dir   = [dir for dir in models_dir if 'crisp' in dir][0]
    crisp_model = func.read_binary(crisp_dir,data_type=data_format,shape=(Nz,Ny,Nx)) # read model
    crisp_model = np.flip(crisp_model,axis=1)
    print('+',crisp_dir)

    # Fuzzy model
    Nm = len(models_dir)-1 # number of fuzzy model
    global func_read
    def func_read(i):
        tag = data_type[data_type['idx']==i]['type_'+str(type)].values[0]
        for dir in models_dir:
            if tag in dir: model_dir = dir
        model = func.read_binary(model_dir,data_type=data_format,shape=(Nz,Ny,Nx))
        print('+',model_dir)
        return (i,model)
    pool = multiprocessing.Pool(ac)
    fuzzy_models = pool.map(func_read,range(Nm))
    pool.close()

    fuzzy_models = sorted(fuzzy_models,key=operator.itemgetter(0))
    fuzzy_models = [i[1] for i in fuzzy_models]
    fuzzy_models = np.stack(fuzzy_models)
    fuzzy_models = np.flip(np.moveaxis(fuzzy_models,0,-1),axis=1)

    # Normalization ([0,1])
    fuzzy_models = (fuzzy_models-range_min)/(range_max-range_min)

    print('> Discrete model ([Nz,Ny,Nx] = {})'.format(crisp_model.shape))
    print('> Fuzzy model ([Nz,Ny,Nx,Nc] = {})'.format(fuzzy_models.shape))
    return crisp_model, fuzzy_models

def model2noisymodel(models,fluctuation_percent=0.1):
    print('Model to noisy model ...')
    # sigma = models*fluctuation_percent
    sigma = 0.1
    noise = np.random.normal(loc=0.0,scale=sigma)
    noise = np.where(noise<-2.0*sigma,0.0,noise)
    noise = np.where(noise> 2.0*sigma,0.0,noise)
    models_noisy = models + noise
    models_noisy = np.maximum(models_noisy,0.0)
    mean  = np.sum(models_noisy,axis=-1,keepdims=True)
    models_noisy = models_noisy/mean
    return models_noisy

def model2image(sub_name,model,tau,fluctuation='True',fraction=0.03,ac=4,random_seed=0):
    """
    Convert fuzzy models into T2 weighted images without noise.
    #### ARGUMENTS
    - model, fuzzy models   [Nz,Nx,Ny,Nm].
    - tes,   echo times     [Nq]. (ms)

    #### RETURN
    - imgs, weighted images without noise.
    """
    Nz,Ny,Nx,Nm = model.shape
    type  = sub_info[sub_info['subject_name']==sub_name]['type'].values[0]
    p_ref = tissue_para[tissue_para['tissue_type']=='csf']
    m0_mu_ref, m0_sigma_ref = p_ref['m0_mu'].values[0], p_ref['m0_sigma'].values[0]

    print('Model to weighted images ...')
    print('> Number of samples :',Nz)
    # Models to weighted images.
    global func_relax_m
    if fluctuation == False:
        M0 = np.zeros(shape=Nm)
        T2 = np.zeros(shape=Nm)
        def func_relax_m(i):
            RandGen = np.random.RandomState(random_seed+i)
            M0_ref = RandGen.normal(loc=m0_mu_ref, scale=m0_sigma_ref) # get CSF M0 value.
            M0_ref = np.minimum(np.maximum(M0_ref, m0_mu_ref-2.0*m0_sigma_ref),m0_mu_ref+2.0*m0_sigma_ref)
            for j in range(Nm):
                p = tissue_para[tissue_para['tissue_type'] == (data_type[data_type['idx']==j]['type_'+str(type)].values[0])]
                t2_mu, t2_sigma = p['t2_mu'].values[0], p['t2_sigma'].values[0]
                M0[j] = M0_ref*p['m0_mu'].values[0]
                t2    = RandGen.normal(loc=t2_mu,scale=t2_sigma)
                T2[j] = np.minimum(np.maximum(t2,t2_mu-2.0*t2_sigma),t2_mu+2.0*t2_sigma)
            # Relaxometry
            frac = model[i]
            s    = frac[...,1:,np.newaxis]*M0[1:,np.newaxis]*np.exp(-tau/T2[1:,np.newaxis]) # without bkg part
            s    = np.sum(s,axis=-2)
            return (i,s)

    if fluctuation == True:
        map_shape=(Ny,Nx)
        M0   = np.zeros(shape=(Ny,Nx,Nm))
        T2   = np.zeros(shape=(Ny,Nx,Nm))
        def func_relax_m(i):
            RandGen = np.random.RandomState(random_seed+i)
            # RandGen = np.random.RandomState()
            M0_ref = RandGen.normal(loc=m0_mu_ref, scale=m0_sigma_ref) # get CSF M0 value.
            M0_ref = np.where(M0_ref<m0_mu_ref-2.0*m0_sigma_ref,m0_mu_ref,M0_ref)
            M0_ref = np.where(M0_ref>m0_mu_ref+2.0*m0_sigma_ref,m0_mu_ref,M0_ref)

            M0_ref_map = RandGen.normal(loc=M0_ref,scale=M0_ref*fraction,size=map_shape)
            M0_ref_map = np.where(M0_ref_map<M0_ref-2.0*M0_ref*fraction,M0_ref,M0_ref_map)
            M0_ref_map = np.where(M0_ref_map>M0_ref+2.0*M0_ref*fraction,M0_ref,M0_ref_map)

            for j in range(Nm):
                p = tissue_para[tissue_para['tissue_type'] == (data_type[data_type['idx']==j]['type_'+str(type)].values[0])]
                t2_mu, t2_sigma = p['t2_mu'].values[0], p['t2_sigma'].values[0]
                M0[...,j] = np.maximum(M0_ref_map*p['m0_mu'].values[0],0.0)

                t2 = RandGen.normal(loc=t2_mu,scale=t2_sigma)
                t2 = np.where(t2<t2_mu-2.0*t2_sigma,t2_mu,t2)
                t2 = np.where(t2>t2_mu+2.0*t2_sigma,t2_mu,t2)

                t2_map = RandGen.normal(loc=t2,scale=t2*fraction,size=map_shape)
                t2_map = np.where(t2_map<t2-2.0*t2*fraction,t2,t2_map)
                t2_map = np.where(t2_map>t2+2.0*t2*fraction,t2,t2_map)
                T2[...,j] = np.maximum(t2_map,0.0)
            # Relaxometry
            frac = model[i]
            s    = frac[...,1:,np.newaxis]*M0[...,1:,np.newaxis]*np.exp(-tau/T2[...,1:,np.newaxis]) # without bkg part
            s    = np.sum(s,axis=-2)
            return (i,s)

    imgs = []
    pool = multiprocessing.Pool(ac)
    pbar = tqdm.tqdm(total=Nz,desc='Model2Image',leave=True)
    for img in pool.imap_unordered(func_relax_m,range(Nz)):
        imgs.append(img)
        pbar.update(1)
    pool.close()
    pbar.close()
    
    imgs = sorted(imgs,key=operator.itemgetter(0))
    imgs = [i[1] for i in imgs]
    imgs = np.stack(imgs)
    print('> Weighted images (multi-exp): '+str(imgs.shape))
    return imgs

def model_to_T1w_image(sub_name,model,tau,fraction=0.025,ac=4,random_seed=0):
    """
    Convert BrainWeb fuzzy models into T1 weighted images without noise.
    #### ARGUMENTS
    - model, BrainWeb fuzzy models [Nz,Ny,Nx,Nm], where Nz is the number of samples, Ny, rows, Nx, columes, Nm, the number of tissues types.
    - tau, inversion times (ms) [Nq], where Nq is the number of inversion times.
    - fraction, intra-tissue variation, percentage of the parameter value.

    #### RETURN
    - imgs, weighted images without noise. [Nz,Ny,Nx,Nq]
    """
    Nz,Ny,Nx,Nm = model.shape
    # load tissue parameter data
    type  = sub_info[sub_info['subject_name']==sub_name]['type'].values[0]
    p_ref = tissue_para[tissue_para['tissue_type']=='csf']
    m0_mu_ref, m0_sigma_ref = p_ref['m0_mu'].values[0], p_ref['m0_sigma'].values[0]

    print('Convert model to T1 weighted images ...')
    print('> Number of samples :',Nz)
    map_shape=(Ny,Nx)

    global func_relax_m
    def func_relax_m(i):
        # Convert model into M0 and T1 maps
        # Reference M0 map.
        RandGen = np.random.RandomState(random_seed+i) # random number generator
        M0_ref  = RandGen.normal(loc=m0_mu_ref, scale=m0_sigma_ref) # get CSF M0 value.
        # limite the value to [mean-2*sigma, mean+2*sigma].
        M0_ref  = np.where(M0_ref<m0_mu_ref-2.0*m0_sigma_ref,m0_mu_ref,M0_ref)
        M0_ref  = np.where(M0_ref>m0_mu_ref+2.0*m0_sigma_ref,m0_mu_ref,M0_ref)
        # add intra-tissue variation.
        M0_ref_map = RandGen.normal(loc=M0_ref,scale=M0_ref*fraction,size=map_shape)
        M0_ref_map = np.where(M0_ref_map<M0_ref-2.0*M0_ref*fraction,M0_ref,M0_ref_map)
        M0_ref_map = np.where(M0_ref_map>M0_ref+2.0*M0_ref*fraction,M0_ref,M0_ref_map)

        M0 = np.zeros(shape=(Ny,Nx,Nm))
        T1 = np.zeros(shape=(Ny,Nx,Nm))

        for j in range(Nm):
            # get the parameters of tissue i
            p = tissue_para[tissue_para['tissue_type'] == (data_type[data_type['idx']==j]['type_'+str(type)].values[0])]
            t1_mu, t1_sigma = p['t1_mu'].values[0], p['t1_sigma'].values[0]
            m0_mu = p['m0_mu'].values[0]

            m0_map = M0_ref_map*m0_mu

            t1 = RandGen.normal(loc=t1_mu,scale=t1_sigma)
            t1 = np.where(t1<t1_mu-2.0*t1_sigma,t1_mu,t1)
            t1 = np.where(t1>t1_mu+2.0*t1_sigma,t1_mu,t1)

            t1_map = RandGen.normal(loc=t1,scale=t1*fraction,size=map_shape)
            t1_map = np.where(t1_map<t1-2.0*t1*fraction,t1,t1_map)
            t1_map = np.where(t1_map>t1+2.0*t1*fraction,t1,t1_map)

            M0[...,j] = np.maximum(m0_map,0.0)
            T1[...,j] = np.maximum(t1_map,0.0)
        
        # set flip angel (FA) and TR value.
        FA = RandGen.uniform(low=2.0,high=20.0,size=1)
        TR = RandGen.uniform(low=5,high=50,size=1)
        
        # covert M0 and T1 maps into A, B, and T1* maps.
        T1s = func.T1_to_T1s(T1=T1,FA=FA,TR=TR) # para(3)
        np.seterr(divide='ignore',invalid='ignore')
        M0s = np.divide(M0*T1s,T1,out=np.zeros_like(T1),where=T1!=0)
        A = M0s # para(1)
        B = M0s+M0 # para(2)

        # relaxometry using three parameter fitting model, A-Bexp(-TI/T1*).
        frac = model[i]
        s = frac[...,1:,np.newaxis]*(A[...,1:,np.newaxis]-B[...,1:,np.newaxis]*np.exp(-tau/T1s[...,1:,np.newaxis])) # without bkg part
        s = np.sum(s,axis=-2)
        return (i,s)
    # parallel processing each patch.
    imgs = []
    pool = multiprocessing.Pool(ac)
    pbar = tqdm.tqdm(total=Nz,desc='Models to T1w images',leave=True)
    for img in pool.imap_unordered(func_relax_m,range(Nz)):
        imgs.append(img)
        pbar.update(1)
    pool.close()
    pbar.close()
    # sort patches according to patch index.
    imgs = sorted(imgs,key=operator.itemgetter(0))
    imgs = [i[1] for i in imgs] # only get patch data.
    imgs = np.stack(imgs)
    print('> T1-weighted images (multi-exp): '+str(imgs.shape))
    return imgs

def image2map(imgs,tau,fitting_model='exp_mono',signed=True,parameter_type='time',algorithm='LL',pbar_disable=False,ac=1):
    '''
    - imgs, weighted MR images. [N,Ny,Nx,Nq]
    - tau, TE or TI.
    - parameter_type, `time`, output relaxation time, `rate`, output relaxation rate.
    - algorithm, `LL`, log-linear fitting algorithm; `NLLS`, non-linear least square fitting algorihtm.
    - pbar_disable, disable the fitting processing bar.
    - ac, parallel processing.
    '''
    print('Weighted images to parameter maps...')
    N,_,_,_ = imgs.shape
    if algorithm == 'NLLS':
        print('> Pixel-wise fitting...')
        maps = []
        pbar = tqdm.tqdm(total=N,desc='Weighted images to parameter maps',leave=True)
        for i in range(N):
            map = func.PixelWiseMapping(imgs=imgs[i],tau=tau,fitting_model=fitting_model,signed=signed,parameter_type=parameter_type,
                                        pbar_disable=pbar_disable,pbar_leave=False,ac=ac)
            maps.append(map)
            pbar.update(1)
        pbar.close()
        maps = np.stack(maps)
        print('')

    if algorithm == 'LL':
        print('> Log-linear fitting...')
        maps = func.LogLinear(s=imgs,tau=tau,n=0,parameter_type=parameter_type)
    print('> Parameter maps (gt): '+ str(maps.shape))
    return maps

def map2image(maps,tau,type_para='T2',signed=True):
    """
    Parameter maps to weigthed images (mono-exponential).
    #### ARGUMENTS
    - maps, parameter maps. [N,Ny,Nx,Np]
    - tau,  echo times or inversion times. [Nq]
    
    #### RETURN
    - imgs, weigthed images. [N,Ny,Nx,Nq]
    """
    print('{} Map to weighted image ...'.format(type_para))
    N,Ny,Nx,Np = maps.shape
    Nq   = tau.shape[-1]
    imgs = np.zeros(shape=(N,Ny,Nx,Nq))
    pbar = tqdm.tqdm(desc='Relax',total=N,leave=True)
    # T2 relaxation
    if type_para == 'T2':
        for n in range(N):
            pbar.update(1)
            imgs[n] = func.Relax_T2(A=maps[n,...,0],T2=maps[n,...,1],tau=tau)
    # T1 relaxation
    if type_para == 'T1':
        for n in range(N):
            pbar.update(1)
            imgs[n] = func.Relax_T1_three_para_magn(A=maps[n,...,0],B=maps[n,...,1],T1=maps[n,...,2],tau=tau,signed=True)
    pbar.close()
    print('> imgs (nf): '+ str(imgs.shape))
    return imgs

def time_rate_convert(maps):
    maps_convert = np.zeros_like(maps)
    maps_convert[...,0] = maps[...,0]
    maps_convert[...,1] = np.where(maps[...,1]==0.0,0.0,1.0/maps[...,1])
    return maps_convert

def image2noisyimage(imgs,sigma,noise_type='Gaussian',NCoils=1):
    """
    Add noise to images.
    #### ARGUMNETS
    - imgs, image without noise. [N,Ny,Nx,Nq]
    - sigma, noise satndard deviation.

    #### RETURN
    - imgs_n, image with noise.
    """
    print('Add noise ...')
    if type(sigma) == list or sigma is None:
        imgs_n = func.addNoiseMix(imgs=imgs,sigma_low=min(sigma),sigma_high=max(sigma),noise_type=noise_type,NCoils=NCoils)
    if type(sigma) == float or type(sigma) == int:
        imgs_n = func.addNoise(imgs=imgs,sigma=sigma,noise_type=noise_type,NCoils=NCoils)
    return imgs_n

def maskBKG(fuzzy_models, crisp_model):
    print('Mask out background ...')
    mask = np.where(crisp_model>0.0,1.0,0.0)
    mask = mask[...,np.newaxis]
    mask = np.repeat(mask,repeats=fuzzy_models.shape[-1],axis=-1)
    fuzzy_models_masked = fuzzy_models*mask
    return fuzzy_models_masked

