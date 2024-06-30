import numpy as np
import matplotlib.pyplot as plt

tes = np.array([10.0,20.0,40.0,80.0,160.0,320.0])
# tes = np.array([5.0,15.0,35.0,75.0,455.0,315.0])
# tes = np.arange(start=5.0,stop=165.0,step=5.0)
# tes = np.arange(start=10.0,stop=330.0,step=10.0)
tes = tes/100.0

p = np.array([10.0, 66.7, 100.0, 200.0, 300.0, 2000.0])
p = 100.0/p

print(p)
print('='*20)

def exp(tau,p):
    return np.exp(-tau*p)

def dfdpp(tau,p,b):
    d = 2.0*(tau**2)*np.exp(-2.0*tau*p) - (tau**2)*b*np.exp(-tau*p)
    d = np.sum(d)
    return d

def dfdp(tau,p,b):
    d = -tau*np.exp(-2.0*tau*p)+tau*b*np.exp(-tau*p)
    # d = np.sum(d)
    d = np.mean(d)
    return d

def drdp(tau,p,d):
    d = -1.0*tau*np.exp(-1.0*tau*p)
    d = np.sum(d)
    return d

num_grid = 200
p_ini = 0.0
p_max = 11.0
p_min = 0.001

p_step = (p_max-p_min)/num_grid
p_grid = np.arange(start=p_min,stop=p_max,step=p_step)
e_grid = np.zeros_like(p_grid)

Nk = 30
ps = np.zeros(shape=Nk,dtype=np.float)
pse= np.zeros(shape=Nk,dtype=np.float)
ds= np.zeros(shape=Nk,dtype=np.float)

Np = p.shape[-1]
fig,axes = plt.subplots(nrows=1,ncols=Np,figsize=(Np*3,3),dpi=300,tight_layout=True)

for k in range(Np):
    s = exp(tes,p[k])
    # b = s+np.random.normal(loc=0.0,scale=0.05,size=s.shape)
    b = s

    for i in range(num_grid):
        se = exp(tes,p_grid[i])
        e_grid[i] = np.sum((s-se)**2)
    axes[k].plot(p_grid*10.0,e_grid)

    pn     = p_ini
    for n in range(Nk):
        g = dfdp(tes,pn,b)

        # h = dfdpp(tes,pn,b)
        # h = h + 0.001

        h = drdp(tes,pn,b)
        h = h**2
        # h = h + 0.001

        # d = g/h
        d = 1.0*g
        pn = pn - d

        ds[n]  = d
        ps[n]  = pn
        pse[n] = np.sum((exp(tes,pn)-s)**2)
    
    for j in range(Nk):
        axes[k].plot(ps[j]*10.0,pse[j],'o')
    
    axes[k].plot(p_ini*10.0,np.sum((exp(tes,p_ini)-s)**2),'k*')
    axes[k].plot(p[k]*10.0,np.sum((exp(tes,p[k])-s)**2),'r*')

    print(ds)
    print(ps)
    print('-'*10)

plt.savefig('figures/tmp')


