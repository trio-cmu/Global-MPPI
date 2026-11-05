"""One-shot Wendland Kernel Surrogate GNC on a 1D curve reconstructed from a user reference.
The surrogate is a Nadaraya-Watson average with the compactly supported C^2 Wendland
kernel (1 - r)^4 (4 r + 1), r = |u - u_i| / support: only samples within the support
radius count, and the support shrinks over the GNC stages.
The three narrow valleys are an analytical approximation of the BLACK trace;
not pixel-perfect digitization and not a MuJoCo measurement.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

OUT = Path('/mnt/data/black_curve_kernel_gnc')
OUT.mkdir(exist_ok=True, parents=True)

# 0 <= x <= 1. Black reference = flat high cost and three sharp valleys.
x_dense = np.linspace(0, 1, 6001)

def triangle(x, center, half_width):
    return np.maximum(1.0 - np.abs(np.asarray(x)-center)/half_width, 0.)

# Approximate three dips in the supplied BLACK curve, not its red traces.
raw = (1.0
       - 0.78 * triangle(x_dense, 0.266, 0.057)
       - 0.38 * triangle(x_dense, 0.520, 0.058)
       - 1.02 * triangle(x_dense, 0.776, 0.058))
# Light pixel-equivalent rounding (the reference black tips have tiny roundness)
y_dense = gaussian_filter1d(raw, sigma=5.5, mode='nearest')

def true_function(x):
    return np.interp(x, x_dense, y_dense)

# Single one-shot cost dataset, used unchanged for all GNC stages.
N = 101
xs = np.linspace(0, 1, N)
ys = true_function(xs)


def wendland(r):
    """C^2 Wendland kernel, positive definite up to 3D: (1 - r)^4 (4 r + 1) for r < 1, else 0."""
    r = np.minimum(r, 1.0)
    return (1 - r)**4 * (4*r + 1)


def nw_kernel(xquery, support):
    """Nadaraya-Watson average of the samples with Wendland weights of this support radius."""
    q = np.atleast_1d(xquery)
    w = wendland(np.abs(q[:, None] - xs[None, :]) / support)
    return (w @ ys) / np.sum(w, axis=1)

# Support radii (about 2.8x the Gaussian bandwidths 0.18, 0.065, 0.012, so the smoothing is
# similar); the smallest must exceed half the sample spacing (0.005) so every query has a sample.
bandwidths = [0.5, 0.18, 0.035]
curves = [nw_kernel(x_dense, h) for h in bandwidths]
colors = ['#53769A','#CF902B','#148F84']

# An original-only plot that uses no red lines.
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10})
fig, ax = plt.subplots(figsize=(10,3.5), dpi=170)
ax.plot(x_dense,y_dense,color='#1A1A1A',lw=2.8,label='Reconstructed black reference')
ax.set(xlabel='1D control variable  u',ylabel='Original cost  J(u)', xlim=(0,1),ylim=(-0.19,1.13))
ax.set_title('Original 1D cost: three narrow valleys (black trace only)',fontweight='bold',loc='left',pad=14)
ax.grid(alpha=.13);ax.spines[['top','right']].set_visible(False)
fig.tight_layout()
fig.savefig(OUT/'black_original_only.png',bbox_inches='tight')
plt.close(fig)

# Four-panel progression
fig,axes=plt.subplots(2,2,figsize=(12,7.2),sharex=True,sharey=True,dpi=170)
axes=axes.flat
ax=axes[0]
ax.plot(x_dense,y_dense,color='#202020',lw=2.5,label='Original black curve')
ax.scatter(xs,ys,color='#3E78B2',s=15,zorder=3,label=f'{N} one-shot evaluations',edgecolors='white',linewidths=.3)
ax.set_title(f'A   Original black curve + {N} observations',loc='left',weight='bold',pad=10)
ax.legend(loc='lower left',fontsize=8,framealpha=.9)
for i,(h,curve,c) in enumerate(zip(bandwidths,curves,colors),start=1):
    ax=axes[i]
    ax.plot(x_dense,y_dense,color='#202020',lw=1.9,alpha=.7,label='Original (reference)')
    ax.plot(x_dense,curve,color=c,lw=2.65,label=f'Wendland surrogate: support = {h:g}')
    ax.scatter(xs,ys,color='grey',s=8,alpha=.25,zorder=4)
    i_min=int(np.argmin(curve))
    ax.scatter([x_dense[i_min]],[curve[i_min]],marker='*',s=145,color=c,edgecolor='white',linewidth=.5,zorder=5)
    ax.set_title(f'{"BCD"[i-1]}   GNC stage {i}  |  support = {h:g}  |  +0 rollouts',loc='left',weight='bold',pad=10)
    ax.legend(loc='lower left',fontsize=8,framealpha=.9)
for ax in axes:
    ax.grid(alpha=.12)
    ax.spines[['top','right']].set_visible(False)
    ax.set_xlim(0,1)
    ax.set_ylim(-.23,1.13)
    ax.set_ylabel('Cost')
axes[2].set_xlabel('Control variable u')
axes[3].set_xlabel('Control variable u')
fig.suptitle('One-shot Wendland Kernel Surrogate GNC | only the BLACK reference curve',fontsize=15,weight='bold',y=.99)
fig.text(.5,.016,f'Same {N} original cost evaluations in all stages; no additional true-cost evaluations to change the support radius.  Synthetic reconstruction, not MuJoCo data.',
         ha='center',fontsize=8.5,color='#545F69')
fig.subplots_adjust(left=.08,right=.98,top=.91,bottom=.095,wspace=.16,hspace=.28)
fig.savefig(OUT/'black_reference_kernel_gnc_stages.png',bbox_inches='tight')
plt.close(fig)

# Overlay figure
fig,ax=plt.subplots(figsize=(11,4.5),dpi=170)
ax.plot(x_dense,y_dense,c='#202020',lw=2.5,label='Original black curve')
for h,curve,c in zip(bandwidths,curves,colors):
    ax.plot(x_dense,curve,c=c,lw=2.1,label=f'Wendland surrogate, support={h:g}')
ax.scatter(xs,ys,c='grey',s=11,alpha=.35,zorder=10,label=f'{N} fixed samples')
ax.set(xlabel='Control variable u',ylabel='Cost',xlim=(0,1),ylim=(-.24,1.12))
ax.set_title('Same observed data; only the Wendland support radius changes',loc='left',weight='bold',pad=12)
ax.spines[['top','right']].set_visible(False)
ax.grid(alpha=.12)
ax.legend(loc='lower left',ncol=2,fontsize=9,framealpha=.9)
fig.tight_layout()
fig.savefig(OUT/'black_reference_kernel_gnc_overlay.png',bbox_inches='tight')
plt.close(fig)

np.savetxt(OUT/'observations.csv',np.c_[xs,ys],delimiter=',',header='u,cost',comments='')
np.savetxt(OUT/'curves.csv',np.c_[x_dense,y_dense,*curves],delimiter=',',header='u,original_reference,' + ','.join(f'surrogate_support_{h:g}' for h in bandwidths),comments='')
for h,c in zip(bandwidths,curves):
    j=np.argmin(c);print('support=',h,'argmin=',x_dense[j],'pred=',c[j],'original_cost=',y_dense[j])
print('Original global min:',x_dense[np.argmin(y_dense)],y_dense.min())
print('Output:',OUT)
