"""10D cost-aware sequential sampling with a certified aggregate chordal sparsity bound.

The SDP assumption is M(alpha)=lam*I+K diag(alpha) K.T, so its aggregate
pattern is Boolean(K @ K.T), i.e., the closed-neighborhood graph square.

The constructive chordal certificate H is maintained by admitting a new sample
only when its entire two-hop neighborhood in the *existing* kernel graph
is already a clique of H and contains at most MAX_CLIQUE-1 nodes.
"""
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import qmc
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee, connected_components

SEED=20261028
DIM=10; N=1000; SUPPORT=.569; MAX_CLIQUE=20
POOL_POW=16; POOL_SIZE=2**POOL_POW
rng=np.random.default_rng(SEED)
# Five well-separated centers, fixed randomly, reproducible
centers=qmc.Sobol(DIM, scramble=True,seed=411).random_base2(3)[:5]
biases=np.array([0,.085,.17,.24,.30])
scales=np.array([1.,1.10,.90,1.08,.95])
# All dimensions matter, mild cross-dimension coupling makes landscape non-axis-aligned
B = rng.normal(size=(DIM,DIM))
U,_,V=np.linalg.svd(B)
rot=U@V

def objective(x):
    """Five-basin 10D nonconvex objective; low is good."""
    x=np.atleast_2d(x)
    # main modes: squared distance to centers with mild rotated anisotropy
    dif=(x[:,None,:]-centers[None,:,:])
    z=dif@rot
    energy=np.sum(dif*dif,axis=-1)+.22*np.sum((z*z)*np.linspace(.5,1.5,DIM),axis=-1)
    c=energy*scales[None,:]+biases[None,:]
    # smooth ripples for local structure without dominating the basins
    osc=.025*(np.sin(9*x[:,0]+5*x[:,1])**2+np.sin(8*x[:,4]-4*x[:,6])**2)
    return c.min(axis=1)+osc

def get_patterns(X,h=SUPPORT):
    D=cKDTree(X).sparse_distance_matrix(cKDTree(X),h,output_type='coo_matrix')
    # ball search may include diagonal, explicitly construct adjacency
    mask=(D.row!=D.col) & (D.data<h)
    kernel=csr_matrix((np.ones(mask.sum(),dtype=np.int8),(D.row[mask],D.col[mask])),shape=(len(X),len(X)))
    kernel=((kernel+kernel.T)>0).astype(np.int8)
    eye=csr_matrix(np.eye(len(X),dtype=np.int8))
    with_diag=(kernel+eye).astype(np.int8)
    aggregate=(with_diag @ with_diag.T).astype(bool).astype(np.int8)
    aggregate.setdiag(0);aggregate.eliminate_zeros()
    return kernel,aggregate

def stats(X):
    k,a=get_patterns(X)
    deg=np.asarray(k.sum(axis=1)).ravel()
    aggdeg=np.asarray(a.sum(axis=1)).ravel()
    cc,_=connected_components(k,directed=False)
    return {'kernel_edges':int(k.nnz//2),'aggregate_edges':int(a.nnz//2),
            'isolated':int(np.sum(deg==0)),'components':int(cc),
            'max_closed_kernel_neighborhood':int(deg.max()+1),
            'max_aggregate_degree':int(aggdeg.max())}, k, a

def sample_with_certificate(pool,cost,accept_target=N):
    # ordered best-to-worst cost; no ball hierarchy, no forced clusters
    order=np.argsort(cost,kind='stable')
    x_arr=np.empty((accept_target,DIM))
    idx_arr=np.empty(accept_target,dtype=np.int32)
    G=[];H=[]
    tree=None; tree_size=0; current=0
    reject={'clique_overflow':0,'not_clique_in_H':0}
    start=time.monotonic()
    for ci in order:
        x=pool[ci]
        if tree is None:
            Nbr=[]
        else:
            Nbr=tree.query_ball_point(x,SUPPORT-1e-12)
        if current>tree_size:
            dif=x_arr[tree_size:current]-x
            dist=np.einsum('ij,ij->i',dif,dif)
            Nbr.extend((np.flatnonzero(dist<SUPPORT**2)+tree_size).tolist())
        # 2-hop neighborhood in *old* kernel graph
        M=set(Nbr)
        for j in Nbr: M.update(G[j])
        if len(M)>=MAX_CLIQUE:
            reject['clique_overflow']+=1; continue
        # M must already form a clique of certificate H
        if any(not (M-{j}).issubset(H[j]) for j in M):
            reject['not_clique_in_H']+=1; continue
        # add new simplicial node to H with neighbor set M
        x_arr[current]=x; idx_arr[current]=ci
        G.append(set(Nbr)); H.append(set(M))
        for j in Nbr: G[j].add(current)
        for j in M: H[j].add(current)
        current+=1
        if current in [100,300,500,700,900,1000]:
            print('Accepted',current,'examined',sum(reject.values())+current,
                  'last cost',round(float(cost[ci]),4),'elapsed',round(time.monotonic()-start,2),flush=True)
        if current%25==0:
            tree=cKDTree(x_arr[:current]);tree_size=current
        if current>=accept_target: break
    return x_arr[:current],idx_arr[:current],G,H,reject

def certificate_check(X,G,H):
    n=len(G)
    # Exact graph-square and chordal-supergraph containment
    ker, agg=get_patterns(X)
    Hs=csr_matrix((np.ones(sum(len(nbr) for nbr in H),dtype=np.int8),
                  (np.fromiter((i for i,ngh in enumerate(H) for j in ngh),dtype=int),
                   np.fromiter((j for ngh in H for j in ngh),dtype=int))),shape=(n,n))
    chk=(agg-Hs).tocoo()
    if np.any(chk.data>0): raise AssertionError('aggregate not contained in H')
    # Reverse insertion order is a perfect elimination ordering because every new node
    # was attached to an existing clique; verify by explicit symbolic elimination.
    alive=[set(q) for q in H]
    max_size=0;fill=0
    for v in range(n-1,-1,-1):
        nbr=list(alive[v]); max_size=max(max_size,1+len(nbr))
        for u in nbr:
            alive[u].remove(v)
        for ai,u in enumerate(nbr):
            for w in nbr[ai+1:]:
                if w not in alive[u]:
                    fill+=1;alive[u].add(w);alive[w].add(u)
        alive[v].clear()
    if fill: raise AssertionError(f'certificate is not chordal; fill={fill}')
    if max_size>MAX_CLIQUE:raise AssertionError('certificate too large')
    return max_size, Hs, agg

def draw_kernel(ax,pat,title,order=None):
    pat = pat if order is None else pat[order,:][:,order]
    ax.spy(pat,markersize=.5,color='#223544')
    ax.set_title(title,fontsize=10)
    ax.set_xlim(-1,pat.shape[0]);ax.set_ylim(pat.shape[0],-1)
    ax.set_xticks([0,500,1000]);ax.set_yticks([0,500,1000])

if __name__=='__main__':
    t0=time.monotonic()
    pool=qmc.Sobol(d=DIM,scramble=True,seed=SEED).random_base2(POOL_POW)
    poolcost=objective(pool)
    uniform=pool[:N].copy();bestcost=pool[np.argsort(poolcost)[:N]].copy()
    X,indices,G,H,rejected=sample_with_certificate(pool,poolcost)
    if len(X)!=N: print('Warning: only',len(X),'accepted; may need more candidates')
    cert,Hmat,A=certificate_check(X,G,H)
    s0,k0,a0=stats(uniform)
    s1,k1,a1=stats(bestcost)
    s2,k2,a2=stats(X)
    # confirm original kernel adjacency exactly equals constructive G
    for i,ng in enumerate(G):
        if set(k2.indices[k2.indptr[i]:k2.indptr[i+1]])!=ng:
            raise AssertionError('Kernel adjacency disagreement')
    print('CERTIFIED max clique:',cert)
    print('CANDIDATES:',len(pool),'accepted:',len(X),'rejected:',rejected)
    for label,XX,s in [('Uniform Sobol',uniform,s0),('Cost only',bestcost,s1),('Constrained',X,s2)]:
        fc=objective(XX)
        print(label,'mean cost',fc.mean(),'median',np.median(fc),'best',fc.min(), 'q90',np.quantile(fc,.9),'graph',s)
    # make cost thresholds based on uniform distribution quantiles
    thresh=np.quantile(objective(uniform),[.1,.25,.5])
    print('Thresholds (Sobol q10,25,50):',thresh)
    for label,XX in [('Uniform',uniform),('Cost-only',bestcost),('Constrained',X)]:
        print(label,'low cost fraction',[(objective(XX)<=t).mean() for t in thresh])
    # chosen insertion order: band-like, and RCM for both patterns
    rcm=reverse_cuthill_mckee(a2,symmetric_mode=True)
    f0=objective(uniform); f1=objective(bestcost); f2=objective(X)
    fig,axs=plt.subplots(2,3,figsize=(14,8.8),layout='constrained')
    draw_kernel(axs[0,0],k2,'Wendland K (selection order)')
    draw_kernel(axs[0,1],a2,'Aggregate SDP (selection order)')
    draw_kernel(axs[0,2],Hmat,'Chordal certificate H (selection order)')
    draw_kernel(axs[1,0],k2,'Wendland K (RCM)',rcm)
    draw_kernel(axs[1,1],a2,'Aggregate SDP (RCM)',rcm)
    draw_kernel(axs[1,2],Hmat,'Chordal certificate H (RCM)',rcm)
    fig.suptitle(f'10D cost-aware sampling | n={len(X)}, h={SUPPORT}, certified max clique={cert}', fontsize=15)
    out='/mnt/data/cost_aware_10d_sparsity.png'
    fig.savefig(out,dpi=190)
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    bins=np.linspace(min(f0.min(),f1.min(),f2.min()),max(f0.max(),f1.max(),f2.max()),45)
    for name,vals in [('Uniform Sobol',f0),('Cost-only',f1),('Chordal-constrained',f2)]:
        axs[0].hist(vals,bins=bins,density=True,alpha=.42,label=name)
    axs[0].set_xlabel('10D objective value (lower is better)')
    axs[0].set_ylabel('Sample density');axs[0].legend()
    labels=['Uniform Sobol','Cost-only','Chordal-constrained']
    axs[1].bar(labels,[f0.mean(),f1.mean(),f2.mean()]);axs[1].set_ylabel('Mean sampled cost (lower is better)')
    axs[1].tick_params(axis='x',rotation=12)
    fig.suptitle('Cost-aware sampling: objective values',fontsize=14)
    fig.savefig('/mnt/data/cost_aware_10d_cost_comparison.png',dpi=190)
    plt.close(fig)
    np.savez_compressed('/mnt/data/cost_aware_10d_results.npz',
                        X_uniform=uniform,X_cost_only=bestcost,X_constrained=X,
                        chosen_candidate_indices=indices,
                        centers=centers,biases=biases,scales=scales,rotation=rot,
                        costs_uniform=f0,costs_cost_only=f1,costs_constrained=f2,
                        kernel_edges=k2.toarray(),aggregate_edges=a2.toarray(),H_chordal=Hmat.toarray(),
                        h=np.array(SUPPORT),clique_max=np.array(cert))
    print('ELAPSED',time.monotonic()-t0)
