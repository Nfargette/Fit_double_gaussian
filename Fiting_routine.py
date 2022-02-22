# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 19:01:59 2022
@author: naisf
"""
import numpy as np
import scipy.stats
import emcee

from matplotlib import rc
from chainconsumer import ChainConsumer
from matplotlib import pyplot as plt

rc('text', usetex=True)
rc('xtick', labelsize=16)
rc('ytick', labelsize=16)

#%%%
#Function definition

def two_gaussians(x, p):
    mu0  = np.array(p[0:2])
    sig0 = np.diag(np.array(p[2:4])**2)
    mu   = np.array(p[4:6])
    sig = np.diag(np.array(p[6:8])**2)
    Gauss0 = scipy.stats.multivariate_normal(mu0, sig0)
    Gauss  = scipy.stats.multivariate_normal(mu, sig)    
    if len(x.shape) >2:
        x_flat = np.array([x[0].flatten(), x[1].flatten()])
        Two_Gauss = lambda x: (1-p[8])*Gauss0.pdf(x) + p[8]*Gauss.pdf(x)
        return Two_Gauss(x_flat.T).reshape(x.shape[1:])
    else:
        return (1-p[8])*Gauss0.pdf(x) + p[8]*Gauss.pdf(x)

def log_prior(p):
    mu_phi0, mu_theta0, sig_phi0, sig_theta0, mu_phi, mu_theta, sig_phi, sig_theta, perc =p
    if (-10 < mu_phi0 < 10. and -10.0 < mu_theta0 < 10. and .1 < sig_phi0 < 30. and .1 < sig_theta0 < 30  
        and -180.0 < mu_phi < 180. and -90.0 < mu_theta < 90. and .1 < sig_phi < 180 and .1 < sig_theta < 90
        and 0.<perc<1.):     
        return 0.0
    return -np.inf

def log_likelihood(p, grid, data):
    Z = two_gaussians(grid, p)
    sigma_e = data.max()*0.1
    S = np.sum((data-Z)**2/sigma_e**2)
    return -S

def log_probability(p, grid, data):
    lp = log_prior(p)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(p, grid, data)    

plt.close('all')

#%%%
enc = 2
#file with phi and theta computed for E2 in radians
#shape 2xn
angles = np.loadtxt('./DATA/E'+str(enc)+'_angles')
angles = angles*180./np.pi

data, x_edges, y_edges = np.histogram2d(angles[0], angles[1], 
                                        bins = (np.arange(-180,181), 
                                                np.arange(-90,91)),
                                        density = 1)
data = data.T
x = (x_edges[1:] + x_edges[:-1])/2.
y = (y_edges[1:] + y_edges[:-1])/2.
X,Y = np.meshgrid(x,y)

#to set to True once run
read_only = True

# backend name
backend_name = 'E' + str(enc) + '.h5'

#initialize walkers
n_walkers = 32

mu_phi_0   = np.random.uniform(-10,10,n_walkers)
mu_theta_0 = np.random.uniform(-10,10,n_walkers)
sig_phi_0   = np.random.uniform(.1,30,n_walkers)
sig_theta_0 = np.random.uniform(.1,30,n_walkers)
mu_phi      = np.random.uniform(-180,180,n_walkers)
mu_theta    = np.random.uniform(-90,90,n_walkers)
sig_phi = np.random.uniform(.1,180,n_walkers)
sig_theta = np.random.uniform(.1,90,n_walkers)
perc = np.random.uniform(0,1,n_walkers)

pos = np.vstack((mu_phi_0, mu_theta_0, sig_phi_0,sig_theta_0,mu_phi,mu_theta,
                 sig_phi,sig_theta,perc)).T
ndim = pos.shape[1]

#run mcmc and save backend if not read_only
if not read_only:
    backend = emcee.backends.HDFBackend(backend_name)
    sampler = emcee.EnsembleSampler(
        n_walkers, ndim, log_probability, backend = backend, 
        args=( np.array([X,Y]), data)
    )
    state = sampler.run_mcmc(pos, 1500, progress=True);
        
reader = emcee.backends.HDFBackend(backend_name, read_only = True)

#plot walker path
fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
samples = reader.get_chain()
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
axes[-1].set_xlabel("step number");

#plot walker position distribution in 9D space
samples = reader.get_chain(discard = 1000,flat = True)

params=[r"$\mu_{0\phi}$", r"$\mu_{0\theta}$",
     r"$\sigma_{0\phi}$", r"$\sigma_{0\theta}$",
     r"$\mu_{\phi}$", r"$\mu_{\theta}$",
     r"$\sigma_{\phi}$", r"$\sigma_{\theta}$",
     r"$\gamma$"]

c = ChainConsumer()
c.add_chain(samples, parameters=params, name = 'Encounter ' + str(enc))
c.configure(summary=True, sigmas=np.linspace(0, 2, 5), shade_gradient=1.5, 
            shade_alpha=.5, cloud = True, num_cloud = 5000, kde = 1.)

results = np.zeros((ndim,3))
for i in range(ndim):
    results[i] = np.array(c.analysis.get_summary()[params[i]])
results= results[:,1]

table = c.analysis.get_latex_table()
print(table)

fig = c.plotter.plot(truth = results, legend=True)
