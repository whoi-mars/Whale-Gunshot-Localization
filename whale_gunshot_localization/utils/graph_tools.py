"""
Code adapted from https://pnnl.github.io/HyperNetX/_modules/algorithms/hypergraph_modularity.html
"""

import numpy as np
import hypernetx.algorithms.hypergraph_modularity as hmod

def last_step(HG, L, wdc=hmod.linear, delta=0.01):
    """
    Given some initial partition L, compute a new partition of the vertices in HG as per Last-Step algorithm [2]_

    Note
    ----
    This is a very simple algorithm that tries moving nodes between communities to improve hypergraph modularity.
    It requires an initial non-trivial partition which can be obtained for example via graph clustering on the 2-section of HG,
    or via Kumar's algorithm.

    Parameters
    ----------
    HG : Hypergraph

    L : list of sets
      some initial partition of the vertices in HG

    wdc : func, optional
        Hyperparameter for hypergraph modularity [2]

    delta : float, optional
            convergence stopping criterion

    Returns
    -------
    : list of sets
      A new partition for the vertices in HG
    """
    A = L[:]  # we will modify this, copy
    D = hmod.part2dict(A)
    qH = 0
    while True:
        for v in list(np.random.permutation(list(HG.nodes))):
            c = D[v]
            s = list(set([c] + [D[i] for i in HG.neighbors(v)]))
            M = []
            if len(s) > 0:
                for i in s:
                    if c == i:
                        M.append(0)
                    else:
                        M.append(
                            hmod._delta_ec(HG, A, v, c, i, wdc)
                            - hmod._delta_dt(HG, A, v, c, i, wdc)
                        )
                i = s[np.argmax(M)]
                if c != i:
                    A[c] = A[c] - {v}
                    A[i] = A[i].union({v})
                    D[v] = i
        Pr = hmod._compute_partition_probas(HG, A)
        q2 = hmod._edge_contribution(HG, A, wdc) - hmod._degree_tax(HG, Pr, wdc)
        if (q2 - qH) < delta:
            break
        qH = q2
    return [a for a in A if len(a) > 0]