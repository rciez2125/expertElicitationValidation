import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import statsmodels.stats.api as sm

fig = plt.figure(figsize=(6,4))
plt.rcParams.update({'font.size': 10})
ax = fig.add_subplot(1,1,1)
fig = sm.GofChisquarePower().plot_power(dep_var = 'nobs', 
                                    nobs = np.arange(2,500), 
                                    effect_size = np.array([0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.25, 0.3, 0.5]), 
                                    #effect_size = np.array([0.15, 0.2, 0.25, 0.5]),
                                    alpha = 0.05, ax = ax, n_bins = 6, ddof = 2, 
                                    title = 'Power of Chi Square'+'\n'+r'$\alpha = 0.05$')
#plt.plot([2, 500], [0.5, 0.5], '--k')
plt.rcParams.update({'legend.loc': 'lower center'})
plt.plot([2, 500], [0.8, 0.8], '--k')
plt.plot([450,450], [0,1], '-k')
#plt.show()
plt.savefig('powerAnalysis_chisquare.png')