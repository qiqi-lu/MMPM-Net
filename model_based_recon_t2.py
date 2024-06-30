"""
Model based compressed sensing method for image reconstruction.

Unconstrained problem
argmin(x) ||A(x)-b||_2^2 + lambda_1*||D(x)||_1

Based on the papers: 
[1] Sparse MRI: The application of compressed sensing for rapid MR imaging.
"""

import os
os.environ["OMP_NUM_THREADS"]        = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"]   = "1" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"]        = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"]    = "1" # export NUMEXPR_NUM_THREADS=6

import numpy as np
import pywt

def A(x,tes):
    """ T2 exponential model.
    """
    m0 = x[...,0][...,np.newaxis]
    t2 = x[...,1][...,np.newaxis]
    t2 = np.where(t2 <= 0.0,1e-7,t2)
    s  = m0*np.exp(-tes/t2)
    return s

def Jr(x,b,tes):
    """ Jacobian*(A(x)-b).
    """
    m0   = x[...,0][...,np.newaxis]
    t2   = x[...,1][...,np.newaxis]
    r    = A(x=x,tes=tes)-b

    # dAdm = np.exp(-tes/(t2+1e-6))
    # dAdt = -tes*m0*dAdm/(t2**2+1e-6)
    t2   = np.where(t2<=0.0,1e-7, t2)
    dAdm = np.exp(-tes/t2)
    dAdt = m0*tes*dAdm/(t2**2)

    Jr_m = np.sum(dAdm*r,axis=-1)
    Jr_t = np.sum(dAdt*r,axis=-1)
    Jr   = np.stack((Jr_m,Jr_t),axis=-1)
    return Jr

def objective_L2(x,b,tes):
    """ Data consistenct term.
    """
    Ax  = A(x,tes)
    obj = (np.linalg.norm(Ax-b))**2
    return obj

def objective_DWT(x,miu=1e-15):
    """ Discrete Wavelet Transform.
    """
    m0 = x[...,0]
    t2 = x[...,1]

    w_m = pywt.wavedec2(m0,'db1',level=4)
    w_t = pywt.wavedec2(t2,'db1',level=4)

    w_m_array,_ = pywt.coeffs_to_array(w_m)
    w_t_array,_ = pywt.coeffs_to_array(w_t)

    obj_l1_m = np.sqrt(np.sum(w_m_array*w_m_array)+miu) 
    obj_l1_t = np.sqrt(np.sum(w_t_array*w_t_array)+miu)

    obj = obj_l1_m + obj_l1_t
    return obj

def grad_L2(x,b,tes):
    """ Gradient of data consistency term.
    """
    grad_l2 = 2*Jr(x=x,b=b,tes=tes)
    return grad_l2

def grad_DWT(x,miu=1e-15):
    """ Gradient of Discrete Wavelet Transform term.
    """
    m0 = x[...,0]
    t2 = x[...,1]

    w_m = pywt.wavedec2(m0,'db1',level=4)
    w_t = pywt.wavedec2(t2,'db1',level=4)

    w_m_array,w_m_slices = pywt.coeffs_to_array(w_m)
    w_t_array,w_t_slices = pywt.coeffs_to_array(w_t)

    grad_l1_m_arr = w_m_array/np.sqrt(w_m_array*w_m_array+miu)
    grad_l1_m_tup = pywt.array_to_coeffs(grad_l1_m_arr,w_m_slices,output_format='wavedec2')
    grad_l1_m     = pywt.waverec2(grad_l1_m_tup,'db1')

    grad_l1_t_arr = w_t_array/np.sqrt(w_t_array*w_t_array+miu)
    grad_l1_t_tup = pywt.array_to_coeffs(grad_l1_t_arr,w_t_slices,output_format='wavedec2')
    grad_l1_t     = pywt.waverec2(grad_l1_t_tup,'db1')

    grad_l1 = np.stack((grad_l1_m,grad_l1_t),axis=-1)
    return grad_l1


def LogLinearN(img,TEs,n=3):
    """
    Log-Linear curve fitting Method.
    Perfrom a pixel-wise linear fit of the decay curve after a log transofrmation (using the first n data point).
    #### AGUMENTS
    - TEs : Echo Time (ms)

    #### RETURN
    - map : parameter maps [M0, T2].
    """
    img = np.abs(img)
    x   = TEs[0:n]
    y   = np.log(img[...,0:n]+1e-5) # log signal

    x_mean = np.mean(x)
    y_mean = np.mean(y,axis=-1,keepdims=True)
    w = np.sum((x-x_mean)*(y-y_mean),axis=-1,keepdims=True)/np.sum((x-x_mean)**2)
    b = y_mean-w*x_mean
    t2  = 1.0/-w[...,0]
    m0  = np.exp(b)[...,0]
    map = np.stack((m0,t2),axis=-1)
    map = np.where(map<0.0,0.0,map)
    map = np.where(map>3000.0,3000.0,map)
    return map

def cs_mri_parameter(b,tes,DC='L2',RE='DWT',lambda_1=0.1,x_init=None,MaxIter=100,scale = 1.0):
    """
    Parameter reconstruction based on compressed sensing.
    #### ARGUMENTS
    - b, measured weighted images, [Ny,Nx,Nc].
    - tes, echo times. (ms)
    - x_init, initial parameter maps. [m0,t2(ms)]
    - lambda_1, data consistency tuning constant.
    - MaxIter, stoping critia by number of iterations.
    - DC, data consistency term.
    - Re, reguularization term.

    #### RETURN
    - x, estimated parameter maps. [m0,t2(ms)]
    """

    dataconsistency = {'L2':objective_L2}
    regularizations = {'DWT':objective_DWT}

    dataconsistency_grad = {'L2':grad_L2}
    regularizations_grad = {'DWT':grad_DWT}

    objective_DC = dataconsistency.get(DC,objective_L2)
    objective_RE = regularizations.get(RE,objective_DWT)

    grad_DC = dataconsistency_grad.get(DC,grad_L2)
    grad_RE = regularizations_grad.get(RE,grad_DWT)

    tes = tes/scale

    # Iniitalization
    if x_init is None:
        map  = LogLinearN(b,tes,n=3) 
        m0 = map[...,0]
        t2 = map[...,1]
        x  = np.stack((m0,t2),axis=-1)
        x  = np.abs(x)
    else:
        x = x_init

    # padding
    Ny,Nx,Nq = b.shape
    Nmax = max(Ny,Nx)
    N  = 2**(np.ceil(np.log2(Nmax)))
    yp = (N-Ny)/2.0
    xp = (N-Nx)/2.0
    yu = int(np.floor(yp))
    yb = int(np.ceil(yp))
    xl = int(np.floor(xp))
    xr = int(np.ceil(xp))
    b = np.pad(b,((yu,yb),(xl,xr),(0,0)),mode='wrap')
    x = np.pad(x,((yu,yb),(xl,xr),(0,0)),mode='wrap')
    print('Padding: ',b.shape)

    # parameters
    TolGrad    = 1e-4   # stopping critia by gradient magnitude
    # line search parameters
    MaxIter_ls = 100
    alpha      = 0.05
    beta       = 0.6 

    # Initialization
    k  = 0
    g0 = grad_DC(x=x,b=b,tes=tes) + lambda_1*grad_RE(x=x)
    dx = -g0
    t0 = 1.0

    while (k <= MaxIter) and (np.linalg.norm(dx) > TolGrad):
        # backtracking line-search
        f0  = objective_DC(x=x,b=b,tes=tes) + lambda_1*objective_RE(x=x)
        t   = t0

        f_DC= objective_DC(x=x+t*dx,b=b,tes=tes)
        f_RE= objective_RE(x=x+t*dx)
        f1  = f_DC + lambda_1*f_RE

        iter_ls = 0
        while (f1 > (f0 + alpha*t*np.sum(g0*dx))) and (iter_ls < MaxIter_ls):
            t  = t * beta
            f1 = objective_DC(x=x+t*dx,b=b,tes=tes) + lambda_1*objective_RE(x=x+t*dx)
            iter_ls = iter_ls+1

        # adaptive initial search step
        if iter_ls == MaxIter_ls: print('Error in line search...')
        if iter_ls > 2: t0 = t0*beta
        if iter_ls < 1: t0 = t0/beta

        # update x
        x = x + t*dx

        # nonlinear conjugate gradient
        g1    = grad_DC(x=x,b=b,tes=tes) + lambda_1*grad_RE(x=x)
        gamma = (np.linalg.norm(g1))**2/(np.linalg.norm(g0))**2 # FR, Fletcher and Reeves (1964)
        # gamma = (np.linalg.norm(g1,axis=(0,1)))**2/(np.sum(dx*(g1-g0),axis=(0,1))+1e-15) # DY, Dai and Yuan (1999)
        # gamma = np.sum(g1*(g1-g0),axis=(0,1))/(-1.0*np.sum(dx*g0,axis=(0,1))+1e-15) # CD, Conjugate Descent (1987)
        g0 =  g1

        # if (k+1)%100==0: 
            # dx = -g1
        # else:
            # dx = -g1 + gamma*dx
        dx = -g1 + gamma*dx

        k  = k + 1
        print('{} | {:>2d} | {:.4e} | {:.4e} | {:.4f} | {:.2f} | {:.2f} | {:.2f}'.format(k,iter_ls,t0,t,gamma,f_DC,f_RE,f1), end='\r')
        # if np.abs(f1-f0)/f0<0.00001: break
    print('')
    print('Total iteration: ',k)

    m0 = x[...,0]
    t2 = x[...,1]*scale # ms
    x  = np.stack((m0,t2),axis=-1)
    x  = x[yu:yu+Ny,xl:xl+Nx,:]
    return x

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    with np.load('test_mri.npz') as data:
        b = data['b']
        tes = data['tes']
        x_gt = data['x_gt']

    x0 = b[...,0:2]*0.0+1.0
    # x0 = b[...,0:2]*0.0
    # x0 = None

    scale = 100.0
    x = cs_mri_parameter(b,tes,DC='L2',RE='DWT',lambda_1=0.05,x_init=x0,MaxIter=200,scale=scale)

    plt.figure(figsize=(18,6),dpi=300)
    col,row = 6,2
    Nq = b.shape[-1]
    for i in range(Nq):
        plt.subplot(row,col,i+1),plt.imshow(b[...,i],cmap='gray',vmin=0.0,vmax=1.2),plt.axis('off'),plt.colorbar(fraction=0.022)
    plt.subplot(row,col,7), plt.imshow(x_gt[...,0],cmap='gray',vmin=0.0,vmax=1.2),plt.axis('off'),plt.colorbar(fraction=0.022)
    plt.subplot(row,col,8), plt.imshow(x_gt[...,1],cmap='jet',vmin=0.0,vmax=300.0),plt.axis('off'),plt.colorbar(fraction=0.022)
    plt.subplot(row,col,9), plt.imshow(x[...,0],cmap='gray',vmin=0.0,vmax=1.2),plt.axis('off'),plt.colorbar(fraction=0.022)
    plt.subplot(row,col,10),plt.imshow(x[...,1],cmap='jet',vmin=0.0,vmax=300.0),plt.axis('off'),plt.colorbar(fraction=0.022)
    plt.subplot(row,col,11),plt.imshow(np.abs(x[...,0]-x_gt[...,0])/(x_gt[...,0]+1e-7),cmap='jet',vmin=0.0,vmax=0.2),plt.axis('off'),plt.colorbar(fraction=0.022)
    plt.subplot(row,col,12),plt.imshow(np.abs(x[...,1]-x_gt[...,1])/(x_gt[...,1]+1e-7),cmap='jet',vmin=0.0,vmax=0.5),plt.axis('off'),plt.colorbar(fraction=0.022)
    plt.savefig('figures/tmp')





