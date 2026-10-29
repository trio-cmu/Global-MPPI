"""Top-10%-seeded, adaptive 10D cost-aware connected chordal-certified sampling.

The initial cost-evaluation budgets are 1k/3k/5k Sobol candidates. Their lowest
10% form an elite navigation pool. Subsequently, local candidates are proposed
around accepted nodes; geometric/chordal feasibility is checked BEFORE paying
for cost evaluation. Acceptance preserves a chordal supergraph H of Boolean K^2.

This is an exploratory benchmark, not an optimality guarantee.
"""
import time, argparse
from collections import Counter
import numpy as np
import networkx as nx
from scipy.stats import qmc
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, eye
from scipy.sparse.csgraph import connected_components, reverse_cuthill_mckee
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cacs_10d_connected_test import f, CENTERS

D=10; N=1000; H_RADIUS=.569; MAX_CLIQUE=20


def prop_batch(X, costs, G, elites, ecost, rng, batch=48):
    """Some near growing tips, some cost-directed toward global elite Sobol locations."""
    t=len(X); rr=H_RADIUS
    # Mix low cost and low degree frontier. Tips give new sparse connectivity.
    deg=np.fromiter((len(s) for s in G), dtype=int)
    c=np.asarray(costs)
    rank=np.argsort(c)
    low_ids=rank[:max(1,int(.35*t))]
    tips=np.flatnonzero(deg<=4)
    if not len(tips): tips=np.arange(t)
    ids=[]
    for v in range(batch):
        r=rng.random()
        if r<.45: ids.append(int(tips[rng.integers(len(tips))]))
        elif r<.85: ids.append(int(low_ids[rng.integers(len(low_ids))]))
        else: ids.append(int(rng.integers(t)))
    ids=np.array(ids)
    anchors=X[ids]
    directions=rng.normal(size=(batch,D))
    directions/=np.maximum(np.linalg.norm(directions,axis=1,keepdims=True),1e-15)
    if len(elites):
        use=rng.random(batch)<.72
        # favor elite positions with lower cost, use random elite to navigate between basins
        ex=rng.integers(min(len(elites),max(10,len(elites)//3)),size=batch)
        targets=elites[ex]
        seek=targets-anchors
        seek=seek/np.maximum(np.linalg.norm(seek,axis=1,keepdims=True),1e-15)
        mix=.84*seek+.54*directions
        mix/=np.maximum(np.linalg.norm(mix,axis=1,keepdims=True),1e-15)
        directions[use]=mix[use]
    step=rr*rng.uniform(.75,.995,size=batch)
    yy=anchors+step[:,None]*directions
    yy=np.clip(yy,0,1)
    return yy,ids


def sample_adaptive(initial_budget,n=N,seed=7301, batch=48,report=False):
    t0=time.perf_counter(); rng=np.random.default_rng(seed+initial_budget)
    initial=qmc.Sobol(D,scramble=True,seed=seed).random_base2(13)[:initial_budget]  # nested Sobol prefix, 8192 generated
    c0=f(initial)
    init_evals=len(c0)
    elite_n=int(np.ceil(.1*initial_budget))
    elite_inds=np.argsort(c0)[:elite_n]
    elites=initial[elite_inds]; ecost=c0[elite_inds]
    X=[elites[0].copy()]; cost=[float(ecost[0])]
    G=[set()];H=[set()]
    rejected=Counter(); sampled=0; extra_cost_evals=0; valid_evals=0
    run_start=time.perf_counter()
    all_eligible=[]
    for _ in range(n-1):
        found=False
        for retry in range(10):
            trial,anchors=prop_batch(np.asarray(X),cost,G,elites,ecost,rng,batch=batch*(retry//3+1))
            sampled+=len(trial)
            Xi=np.asarray(X)
            # Query real 10D kernel neighbors, no approximate assumption
            tree=cKDTree(Xi)
            ss=tree.query_ball_point(trial, r=H_RADIUS-1e-11)
            eligible=[]
            for idx,S_list in enumerate(ss):
                S=set(S_list)
                if not S:
                    rejected['no_neighbor']+=1;continue
                T=set(S)
                for j in S:T.update(G[j])
                if len(T)>=MAX_CLIQUE:
                    rejected['oversize']+=1;continue
                if any(not (T-{j}).issubset(H[j]) for j in T):
                    rejected['not_clique']+=1;continue
                eligible.append((idx,S,T))
            if eligible:
                # Only evaluate the computationally expensive cost on geometrically feasible points
                ids=[idx for idx,_,_ in eligible]
                cc=f(trial[ids]); extra_cost_evals+=len(ids)
                valid_evals+=len(ids)
                # Greedy cost (preserves spread/connectivity only through hard graph constraint)
                ci=int(np.argmin(cc)); idx,S,T=eligible[ci]
                point=trial[idx].copy();val=float(cc[ci]); found=True
                break
        if not found:
            raise RuntimeError(f'No feasible candidate at t={len(X)}; counts={rejected}')
        i=len(X); X.append(point);cost.append(val);G.append(set(S)); H.append(set(T))
        for j in S:G[j].add(i)
        for j in T:H[j].add(i)
        if report and len(X)%200==0:
            print('initial', initial_budget,'n',len(X),'cost',np.mean(cost),'extra cost calls',extra_cost_evals,'elapsed',round(time.perf_counter()-run_start,1),flush=True)
    sample_seconds=time.perf_counter()-run_start
    return {'X':np.asarray(X),'cost':np.asarray(cost),'G':G,'H':H,
            'initial':initial,'initial_cost':c0,'elites':elites,'elite_cost':ecost,
            'budget':initial_budget,'elite_n':elite_n,
            'sampled':sampled,'extra_evals':extra_cost_evals,'total_evals':init_evals+extra_cost_evals,
            'rejected':dict(rejected),'sampling_seconds':sample_seconds,
            'total_seconds':time.perf_counter()-t0}


def graph_matrix(G):
    row=[];col=[]
    for i,S in enumerate(G):
        for j in S:row.append(i);col.append(j)
    return csr_matrix((np.ones(len(row),dtype=np.int8),(row,col)),shape=(len(G),len(G)))


def verify(out):
    X=out['X'];G=out['G'];H=out['H'];n=len(X)
    assert n==1000
    # independently rebuild actual geometry edges
    pair=cKDTree(X).query_pairs(H_RADIUS-1e-11)
    row=[];col=[]
    for i,j in pair:row.extend([i,j]);col.extend([j,i])
    K=csr_matrix((np.ones(len(row),dtype=np.int8),(row,col)),shape=(n,n),dtype=np.int8)
    Gm=graph_matrix(G)
    assert (K!=Gm).nnz==0
    Ki=K.astype(np.int32)+eye(n,format='csr',dtype=np.int32)
    A=((Ki@Ki)>0).astype(np.int8);A.setdiag(0);A.eliminate_zeros()
    Hm=graph_matrix(H)
    assert (A-Hm).maximum(0).nnz==0
    Hg=nx.from_scipy_sparse_array(Hm)
    assert nx.is_chordal(Hg)
    max_c=max(map(len,nx.find_cliques(Hg)))
    assert max_c<=MAX_CLIQUE
    agg=nx.from_scipy_sparse_array(A)
    agg_chordal=nx.is_chordal(agg)
    nc,_=connected_components(K,directed=False)
    assert nc==1
    out.update({'kernel':K,'aggregate':A,'certificate':Hm,'max_clique':max_c,'agg_is_chordal':agg_chordal,
                'components':int(nc),'kernel_edges':K.nnz//2,'aggregate_edges':A.nnz//2,'certificate_edges':Hm.nnz//2})
    return out


def plot_comparison(outputs,path):
    fig,axs=plt.subplots(3,3,figsize=(14,13),layout='constrained')
    for i,out in enumerate(outputs):
        a=out['aggregate'];k=out['kernel'];h=out['certificate'];perm=reverse_cuthill_mckee(a,symmetric_mode=True)
        for j,(M,title) in enumerate([(k,'Kernel'),(a,'Aggregate SDP'),(h,'Chordal certificate')]):
            axs[i,j].spy(M[perm,:][:,perm],markersize=.44,color='#1d665e')
            axs[i,j].set_title(f'M0={out["budget"]:,} (top {out["elite_n"]:,}) | {title}',fontsize=11)
            axs[i,j].set_xlabel('RCM index')
            axs[i,j].set_ylabel('RCM index')
    fig.suptitle('10D adaptive cost-aware CACS | n=1000, h=0.569, max aggregate clique <=20',fontsize=14)
    fig.savefig(path,dpi=155);plt.close(fig)


def plot_metrics(outputs,path):
    fig,axs=plt.subplots(1,3,figsize=(14,4.3),layout='constrained')
    labels=[f'{o["budget"]:,}' for o in outputs]
    axs[0].bar(labels,[o['cost'].mean() for o in outputs],color='#437a75')
    axs[0].set_title('Final mean sample cost (lower better)');axs[0].set_xlabel('Initial Sobol evaluations')
    axs[1].bar(labels,[o['total_evals'] for o in outputs],color='#437a75')
    axs[1].set_title('Total cost evaluations');axs[1].set_xlabel('Initial Sobol evaluations')
    axs[2].bar(labels,[o['sampling_seconds'] for o in outputs],color='#437a75')
    axs[2].set_title('Sequential sampling time (s)');axs[2].set_xlabel('Initial Sobol evaluations')
    fig.suptitle('Top-10% initial proposals: budget / quality trade-off',fontsize=13)
    fig.savefig(path,dpi=165);plt.close(fig)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--n',type=int,default=1000);parser.add_argument('--batch',type=int,default=48);parser.add_argument('--seeds',type=int,default=1)
    args=parser.parse_args()
    outs=[]
    for budget in [1000,3000,5000]:
        print('BEGIN',budget,flush=True)
        out=sample_adaptive(budget,n=args.n,batch=args.batch,report=True)
        if args.n==1000:out=verify(out)
        print('RESULT',budget,'mean',out['cost'].mean(),'min',out['cost'].min(),'cost evals',out['total_evals'], 'extra',out['extra_evals'],'trials',out['sampled'],'runtime',out['sampling_seconds'], 'clique',out.get('max_clique'),flush=True)
        outs.append(out)
    if args.n==1000:
        plot_comparison(outs,'/mnt/data/cacs_top10_initial_pattern_comparison.png')
        plot_metrics(outs,'/mnt/data/cacs_top10_initial_metrics.png')
        records={'budgets':np.array([o['budget'] for o in outs]),'mean_cost':np.array([o['cost'].mean() for o in outs]),
              'total_evals':np.array([o['total_evals'] for o in outs]),'sampling_seconds':np.array([o['sampling_seconds'] for o in outs])}
        for o in outs:
            b=o['budget'];records[f'X_{b}']=o['X'];records[f'cost_{b}']=o['cost']
            records[f'kernel_{b}']=o['kernel'].toarray();records[f'aggregate_{b}']=o['aggregate'].toarray();records[f'certificate_{b}']=o['certificate'].toarray()
        np.savez_compressed('/mnt/data/cacs_top10_initial_benchmark_data.npz',**records)
