"""Budgeted global cost-aware sampling with a certified chordal aggregate SDP graph.

Standalone improvement of cacs_top10_adaptive_benchmark.py.

Improvements:
  (1) A *hard* limit on expensive cost evaluations, and cached Sobol costs.
  (2) Cheap geometric/chordal filtering, then evaluate at most one new point per step.
  (3) Reuse many cost-evaluated initial Sobol points (no re-evaluation).
  (4) Mix local proposals with periodic, independent global Sobol exploration;
      global components are permitted when separated from existing samples.
  (5) Optional delete/reinsert replacement moves, preserving the certificate.
  (6) Independently reconstruct the *actual* Boolean aggregate graph from geometry.

The aggregate graph identity G_agg = G_K^2 assumes the kernel-column SDP form
  M(alpha) = lam*I + K*diag(alpha)*K.T, K the symmetric Wendland matrix.
No lower-bound or optimizer-quality guarantee is claimed.

Usage:
    python cacs_budgeted_global_improved.py --seed 7301 --n 1000 --budget 1200
    python cacs_budgeted_global_improved.py --seed 7301 --n 200 --budget 300 --initial 80
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, eye
from scipy.sparse.csgraph import connected_components, reverse_cuthill_mckee
from scipy.stats import qmc

D = 10
H_RADIUS = 0.569
MAX_CLIQUE = 20

# Identical reproducible synthetic test function to cacs_10d_connected_test.py.
_FUNCTION_RNG = np.random.default_rng(20261028)
CENTERS = qmc.Sobol(D, scramble=True, seed=411).random_base2(3)[:5]
BIAS = np.array([0., .085, .17, .24, .30])
SCALES = np.array([1., 1.10, .90, 1.08, .95])
_T = _FUNCTION_RNG.normal(size=(D,D))
_U, _, _V = np.linalg.svd(_T)
ROT = _U @ _V

def f(X):
    X = np.atleast_2d(np.asarray(X, float))
    dx = X[:,None,:] - CENTERS[None,:,:]
    z = dx @ ROT
    c = ((dx**2).sum(-1) + .22*(z*z*np.linspace(.5,1.5,D)).sum(-1))
    c = c*SCALES[None,:]+BIAS[None,:]
    return c.min(axis=1) + .025*(np.sin(9*X[:,0]+5*X[:,1])**2 + np.sin(8*X[:,4]-4*X[:,6])**2)

class EvaluationBudget:
    """Count expensive *new* evaluations, cache exact previously evaluated points."""
    def __init__(self, max_calls):
        self.limit = int(max_calls)
        self.calls = 0
        self.cache: dict[bytes,float] = {}

    def evaluate(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        keys = [np.ascontiguousarray(row).tobytes() for row in X]
        new = {}
        for key,row in zip(keys,X):
            if key not in self.cache and key not in new:
                new[key] = row.copy()
        if self.calls+len(new) > self.limit:
            raise RuntimeError('Expensive evaluation budget exceeded')
        if new:
            costs = f(np.asarray(list(new.values())))
            self.cache.update(zip(new.keys(), map(float,costs)))
            self.calls += len(new)
        return np.asarray([self.cache[key] for key in keys], dtype=float)

    def cached(self, x):
        return self.cache.get(np.ascontiguousarray(x,dtype=np.float64).tobytes())

class ChordalSampleGraph:
    """G is true geometric kernel graph; H is its chordal aggregate supergraph."""
    def __init__(self, radius=H_RADIUS, max_clique=MAX_CLIQUE):
        self.radius = float(radius)
        self.r = int(max_clique)
        self.X = []
        self.costs = []
        self.G: list[set[int]] = []
        self.H: list[set[int]] = []

    def feasible_from_neighbors(self, neighbors, allow_component=True):
        S = set(map(int, neighbors))
        if not S and not allow_component: return None
        T = set(S)
        for j in S: T.update(self.G[j])
        if len(T) >= self.r: return None
        if any(not (T-{j}).issubset(self.H[j]) for j in T): return None
        return (S,T)

    def feasible(self, y, allow_component=True):
        if not self.X: return (set(),set())
        X = np.asarray(self.X)
        ds = np.linalg.norm(X-np.asarray(y), axis=1)
        return self.feasible_from_neighbors(np.flatnonzero(ds < self.radius-1e-10), allow_component)

    def batch_feasible(self, Y, allow_components=False):
        if not self.X:
            return [(i,(set(),set())) for i in range(len(Y))]
        kd = cKDTree(np.asarray(self.X))
        neighbor_sets = kd.query_ball_point(Y, self.radius-1e-10)
        ans = []
        for i,S in enumerate(neighbor_sets):
            trial = self.feasible_from_neighbors(S,allow_components)
            if trial is not None: ans.append((i,trial))
        return ans

    def append(self, y, cost, st):
        S,T = st
        i = len(self.X)
        self.X.append(np.array(y, dtype=float, copy=True))
        self.costs.append(float(cost))
        self.G.append(set(S))
        self.H.append(set(T))
        for j in S: self.G[j].add(i)
        for j in T: self.H[j].add(i)

    def without(self, i):
        """Deleting arbitrary vertex preserves chordality; old aggregate edges cannot increase."""
        new = ChordalSampleGraph(self.radius, self.r)
        old_ids = [k for k in range(len(self.X)) if k!=i]
        mapping = {old:j for j,old in enumerate(old_ids)}
        new.X = [self.X[k].copy() for k in old_ids]
        new.costs = [self.costs[k] for k in old_ids]
        new.G = [{mapping[j] for j in self.G[k] if j!=i} for k in old_ids]
        new.H = [{mapping[j] for j in self.H[k] if j!=i} for k in old_ids]
        return new

    def components(self):
        n=len(self.X)
        labels=np.full(n,-1,dtype=int)
        t=0
        for v in range(n):
            if labels[v]>=0:continue
            todo=[v];labels[v]=t
            while todo:
                for j in self.G[todo.pop()]:
                    if labels[j]<0:
                        labels[j]=t;todo.append(j)
            t+=1
        return t, labels

def select_initial_seeds(graph, X0, c0, initial_reuse=60):
    """Reuse already evaluated points, prioritize low cost with some spatial diversity."""
    order=np.argsort(c0)
    selected=set()
    # Top 10% are elite guidance; select elites first, then a few cost-diverse seeds.
    for i in order[:max(1,int(np.ceil(.1*len(X0))))]:
        st=graph.feasible(X0[i],allow_component=True)
        if st is not None:
            graph.append(X0[i], c0[i], st); selected.add(int(i))
        if len(selected)>=initial_reuse:break
    # Remaining seeds score = cost + penalty for proximity to already selected points.
    while len(selected)<initial_reuse:
        remaining=np.asarray([j for j in order if int(j) not in selected],dtype=int)
        if not len(remaining):break
        kd=cKDTree(np.asarray(graph.X))
        dist=kd.query(X0[remaining],k=1)[0]
        score=c0[remaining]+.12*np.exp(-.5*(dist/.65)**2)
        progressed=False
        for idx in remaining[np.argsort(score)[:min(50,len(score))]]:
            st=graph.feasible(X0[idx],allow_component=True)
            if st is not None:
                graph.append(X0[idx],c0[idx],st);selected.add(int(idx));progressed=True;break
        if not progressed:break
    return selected

def propose(graph, X0, c0, rng, sobol, batch=96, allow_global=False):
    """Cheap geometric candidates: local cost-biased / degree-biased + independent global."""
    n=len(graph.X)
    X=np.asarray(graph.X)
    cc=np.asarray(graph.costs)
    degrees=np.asarray([len(z) for z in graph.G])
    anchors=np.asarray(X0[np.argsort(c0)[:max(8,int(.1*len(X0)))]])
    # use global points at every iteration, not only first initialization
    n_global=int(batch*.25)
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='The balance properties of Sobol')
        U=sobol.random(n_global)
    Y=np.empty((batch,D))
    Y[:n_global]=U
    w=np.exp(-3.0*(cc-cc.min()))/np.sqrt(1+degrees)
    w=w/w.sum()
    tip=np.flatnonzero(degrees<=3)
    for k in range(n_global,batch):
        mode=rng.random()
        if mode<.20: base=anchors[rng.integers(len(anchors))]
        elif mode<.55 and len(tip):base=X[rng.choice(tip)]
        else:base=X[rng.choice(n,p=w)]
        dr=rng.normal(size=D);dr/=max(np.linalg.norm(dr),1e-12)
        if rng.random()<.20:
            target=anchors[rng.integers(len(anchors))]
            to=target-base
            if np.linalg.norm(to)>1e-9:
                to=to/np.linalg.norm(to)
                dr=.75*to+.48*dr;dr/=max(np.linalg.norm(dr),1e-12)
        step=graph.radius*rng.uniform(.24,1.03)
        Y[k]=np.clip(base+step*dr,0,1)
    return Y,n_global

def rank_candidates(graph, Y, viable, known_X, known_y, exploration=.0):
    """Cheap cost proxy: cost of nearest *accepted* evaluated trajectory.

    This deliberately uses NO synthetic ground-truth costs for untested proposals.
    Empirically more reliable for this 10D small-data test than a 4-NN mean.
    An optional distance term penalizes overly distant extrapolations.
    """
    idx=np.asarray([i for i,_ in viable],int)
    pts=Y[idx]
    dist,near=cKDTree(np.asarray(graph.X)).query(pts,k=1)
    scores=np.asarray(graph.costs)[near] + .35*dist
    return np.argsort(scores)

def sample(initial=200, total_n=1000, budget=1200, seed=7301,
           initial_reuse=40, replacement_evals=40, batch=96,
           global_every=125, max_clique=MAX_CLIQUE, verbose=False):
    """Return complete certified sample set or clearly indicate failure."""
    if budget < initial: raise ValueError('budget must cover initial Sobol evaluations')
    t0=time.perf_counter()
    rng=np.random.default_rng(seed+101)
    sob0=qmc.Sobol(D,scramble=True,seed=seed)
    X0=sob0.random_base2(int(np.ceil(np.log2(initial))))[:initial]
    oracle=EvaluationBudget(budget)
    c0=oracle.evaluate(X0)
    graph=ChordalSampleGraph(max_clique=max_clique)
    selected_initial=select_initial_seeds(graph,X0,c0,min(initial_reuse,total_n))
    # We need enough budget for remaining true evaluations and optional replacement.
    if oracle.calls+total_n-len(graph.X)>budget:
        raise RuntimeError('Insufficient budget for final 1000 true-evaluated samples')
    known_X=[row.copy() for row in X0]
    known_y=list(map(float,c0))
    sob=qmc.Sobol(D,scramble=True,seed=seed+50010)
    # Sobol 32-per-step chunks are not always power-of-two prefixes; this is
    # okay for a proposal pool, not a low-discrepancy guarantee on each chunk.
    trials=Counter(); periodic_global=0
    stalled=0
    while len(graph.X)<total_n:
        if oracle.calls>=budget:
            break
        Y,n_global=propose(graph,X0,c0,rng,sob,batch=batch)
        # Diverse global exploration: allow new disconnected components periodically.
        use_global=(global_every>0 and (len(graph.X) - len(selected_initial))%global_every==0)
        local_idx = np.arange(n_global,len(Y))
        global_idx = np.arange(0,n_global)
        feasible=[]
        # Global points can be inserted as independent components ONLY periodically.
        if use_global:
            fs=graph.batch_feasible(Y[global_idx],allow_components=True)
            feasible.extend([(global_idx[idx],st) for idx,st in fs])
        fl=graph.batch_feasible(Y[local_idx],allow_components=False)
        feasible.extend([(local_idx[idx],st) for idx,st in fl])
        trials['checked']+=len(Y)
        trials['feasible']+=len(feasible)
        if not feasible:
            stalled+=1
            if stalled>=100:break
            continue
        stalled=0
        idx_order=rank_candidates(graph,Y,feasible,np.asarray(known_X),np.asarray(known_y))
        # Once per period ensure a new global *truly independent* exploration point,
        # where possible. This is an optional mechanism for multi-basin discovery.
        if use_global:
            g_candidates=[k for k,(ii,st) in enumerate(feasible) if ii<n_global and not st[0]]
            if g_candidates:
                gsorted=sorted(g_candidates,key=lambda k: np.where(idx_order==k)[0][0])
                chosen=feasible[gsorted[0]];periodic_global+=1
            else:
                chosen=feasible[int(idx_order[0])]
        else:
            chosen=feasible[int(idx_order[0])]
        ii,st=chosen
        y=Y[ii]
        # Graph feasibility was already certified. Exactly ONE new true cost call.
        old_calls=oracle.calls
        val=float(oracle.evaluate(y[None,:])[0])
        trials['new_evals']+=oracle.calls-old_calls
        known_X.append(y.copy()); known_y.append(val)
        graph.append(y,val,st)
        if verbose and len(graph.X)%200==0:
            print(f'n={len(graph.X)} expensive_evals={oracle.calls} mean={np.mean(graph.costs):.4f} components={graph.components()[0]}',flush=True)
    before=float(np.mean(graph.costs)) if graph.X else float('nan')
    # Replacement: drop high-cost sample, cheaply screen candidates, then evaluate ONE.
    successful_replace=0; replacement_attempts=0; replaced_cached=0
    if len(graph.X)==total_n:
        max_tries=min(max(0,int(replacement_evals)),max(0,budget-oracle.calls))
        for it in range(max_tries):
            if oracle.calls>=budget:break
            replacement_attempts+=1
            costs=np.asarray(graph.costs)
            # Select from upper-cost quartile, but vary the target.
            high=np.argsort(costs)[int(.70*len(costs)):]
            old_i=int(high[rng.integers(len(high))]);old_cost=costs[old_i]
            g_minus=graph.without(old_i)
            C,_=propose(g_minus,X0,c0,rng,sob,batch=batch,allow_global=True)
            viable=g_minus.batch_feasible(C,allow_components=True)
            # Also try evaluated unused initial candidates: zero-cost reinsertions.
            cached_candidates=X0[np.argsort(c0)[:100]]
            vc=g_minus.batch_feasible(cached_candidates,allow_components=True)
            viable.extend([(len(C)+j,st) for j,st in vc])
            if not viable:continue
            trial=np.vstack([C,cached_candidates])
            # Prioritize actual cached points with known improving cost.
            good_cached=[(j,st) for j,st in viable
                         if oracle.cached(trial[j]) is not None and oracle.cached(trial[j]) < old_cost-1e-9
                         and min(np.linalg.norm(np.asarray(g_minus.X)-trial[j],axis=1))>1e-8]
            if good_cached:
                jj,st=min(good_cached,key=lambda v:oracle.cached(trial[v[0]]))
                val=float(oracle.cached(trial[jj]));replaced_cached+=1
            else:
                # Use cheap acquisition to choose at most one uncached candidate.
                un=[v for v in viable if oracle.cached(trial[v[0]]) is None]
                if not un:continue
                ranking=rank_candidates(g_minus,trial,un,np.asarray(known_X),np.asarray(known_y))
                jj,st=un[int(ranking[0])]
                val=float(oracle.evaluate(trial[jj][None,:])[0])
                known_X.append(trial[jj].copy());known_y.append(val)
            # Strictly decrease mean sample cost; optional diversity objective can replace this.
            if val < old_cost-1e-8:
                g_minus.append(trial[jj],val,st)
                graph=g_minus
                successful_replace+=1
    out = {
        'X':np.asarray(graph.X),'cost':np.asarray(graph.costs),
        'G':graph.G,'H':graph.H,'initial_X':X0,'initial_cost':c0,
        'seed':seed,'budget':budget,'max_clique_limit':max_clique,'initial_evaluations':initial,
        'initial_reused':len(selected_initial),'evaluations':oracle.calls,
        'selected':len(graph.X),'mean_before_replacement':before,
        'successful_replacements':successful_replace,'replacement_attempts':replacement_attempts,
        'replacements_using_cache':replaced_cached,'global_restarter_points':periodic_global,
        'proposals_checked':trials['checked'],'proposals_feasible':trials['feasible'],
        'runtime_seconds':time.perf_counter()-t0,
        'completed':len(graph.X)==total_n,
    }
    return out

def graph_matrix(sets):
    n=len(sets); row=[];col=[]
    for i,nbr in enumerate(sets):
        for j in nbr: row.append(i);col.append(j)
    return csr_matrix((np.ones(len(row),dtype=np.uint8),(row,col)),shape=(n,n))

def verify(out):
    """Independent geometry / aggregate sparsity / chordality certificate verification."""
    X=out['X'];n=len(X)
    pairs=cKDTree(X).query_pairs(H_RADIUS-1e-10)
    row=[];col=[]
    for i,j in pairs: row.extend((i,j));col.extend((j,i))
    G=csr_matrix((np.ones(len(row),dtype=np.int8),(row,col)),shape=(n,n))
    Gm=graph_matrix(out['G'])
    assert (G!=Gm).nnz==0, 'Kernel edges do not match geometry'
    KD=G.astype(np.int32)+eye(n,format='csr',dtype=np.int32)
    A=((KD@KD)>0).astype(np.int8)
    A.setdiag(0);A.eliminate_zeros()
    H=graph_matrix(out['H']).astype(np.int8)
    assert (A-H).maximum(0).nnz==0, 'Aggregate edges absent from certificate'
    assert (H-H.T).nnz==0, 'Certificate adjacency asymmetric'
    Hnx=nx.from_scipy_sparse_array(H)
    assert nx.is_chordal(Hnx), 'Certificate not chordal'
    maximal=list(nx.find_cliques(Hnx))
    maxcl=max(map(len,maximal),default=0)
    assert maxcl<=out.get('max_clique_limit', MAX_CLIQUE), f'Clique bound failed: {maxcl}'
    comps,_=connected_components(G,directed=False)
    out.update({'kernel_edges':G.nnz//2,'aggregate_edges':A.nnz//2,
                'certificate_edges':H.nnz//2,'max_clique':maxcl,
                'components':int(comps),'fill_edges':(H-A).maximum(0).nnz//2,
                'maximal_cliques':len(maximal),'chordal_actual_agg':nx.is_chordal(nx.from_scipy_sparse_array(A)),
                'G_mat':G,'A_mat':A,'H_mat':H})
    return out

def save_visual(out, dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import cm
    A=out['A_mat'];K=out['G_mat'];H=out['H_mat']
    perm=reverse_cuthill_mckee(A,symmetric_mode=True)
    fig,axs=plt.subplots(2,3,figsize=(13.2,8.1),layout='constrained')
    for i,(M,lab) in enumerate([(K,'Wendland kernel'),(A,'Actual SDP aggregate'),(H,'Chordal certificate')]):
        axs[0,i].spy(M,markersize=.6,color='#207f76')
        axs[0,i].set_title(lab+' | sample order',fontsize=11)
        axs[1,i].spy(M[perm,:][:,perm],markersize=.6,color='#207f76')
        axs[1,i].set_title(lab+' | RCM ordering',fontsize=11)
    fig.suptitle('Budgeted global CACS | 10D, n=%d, h=%.3f | max clique %d | evals %d' % (
        out['selected'],H_RADIUS,out['max_clique'],out['evaluations']))
    fig.savefig(dest,dpi=140);plt.close(fig)

def save_report(out,dest):
    fields=['seed','selected','budget','max_clique_limit','initial_evaluations','initial_reused','evaluations',
            'mean_before_replacement','successful_replacements','replacement_attempts',
            'replacements_using_cache','global_restarter_points','proposals_checked',
            'proposals_feasible','runtime_seconds','completed','kernel_edges','aggregate_edges',
            'certificate_edges','max_clique','components','fill_edges','maximal_cliques','chordal_actual_agg']
    fields_d={key:out[key] for key in fields}
    fields_d['mean_cost']=float(out['cost'].mean());fields_d['best_cost']=float(out['cost'].min())
    with open(dest,'w') as f_:json.dump(fields_d,f_,indent=2)
    return fields_d

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,default=7301)
    p.add_argument('--max-clique',type=int,default=20,help='Use 19 for strict max clique <20')
    p.add_argument('--n',type=int,default=1000)
    p.add_argument('--budget',type=int,default=1200)
    p.add_argument('--initial',type=int,default=200)
    p.add_argument('--reuse',type=int,default=40)
    p.add_argument('--replacement-evals',type=int,default=40)
    p.add_argument('--batch',type=int,default=96)
    p.add_argument('--global-every',type=int,default=125)
    p.add_argument('--outdir',default='/mnt/data')
    args=p.parse_args()
    out=sample(initial=args.initial,total_n=args.n,budget=args.budget,seed=args.seed,
               initial_reuse=args.reuse,replacement_evals=args.replacement_evals,
               batch=args.batch,global_every=args.global_every,max_clique=args.max_clique,verbose=True)
    out=verify(out)
    target=Path(args.outdir);target.mkdir(exist_ok=True,parents=True)
    if out['selected']>=50:
        save_visual(out,target/'cacs_budgeted_improved_patterns.png')
    fields=save_report(out,target/'cacs_budgeted_improved_metrics.json')
    np.savez_compressed(target/'cacs_budgeted_improved_data.npz',
                        X=out['X'],cost=out['cost'],initial_X=out['initial_X'],
                        initial_cost=out['initial_cost'],
                        kernel=out['G_mat'].toarray(),aggregate=out['A_mat'].toarray(),
                        certificate=out['H_mat'].toarray())
    print('RESULT',json.dumps(fields,indent=2),flush=True)

if __name__=='__main__':main()
