# SPDX-FileCopyrightText: 2023 Justin Blanchard <UncombedCoconut@gmail.com>
# SPDX-License-Identifier: Apache-2.0 OR MIT
from collections import deque


def iter_dfas(n, S):
    ''' We wish to solve a TM modulo an equivalence relation on the left-of-head tape configurations.
        This generator yields a list T representing every DFA (Q={0,…,n-1}, |Σ|=S, δ(q,s)=T[S*q+s], q₀=0), such that:
        • T[0]=0 (invariance under leading 0's on the tape),
        • The list [T.index(q) for q in range(n)] is increasing (an arbitrary ordering of the states).'''
    dfa = [0] * (S*n)
    refs = [S] + [0]*(n-1)
    states_used = 1
    while True:
        if states_used == n:
            yield dfa
        for i in reversed(range(1, S*states_used)):
            refs[dfa[i]] -= 1
            if refs[dfa[i]]:
                if dfa[i] < n-1:
                    dfa[i] += 1
                    refs[dfa[i]] += 1
                    if dfa[i] == states_used:
                        states_used += 1
                        refs[0] += S
                    break
                else:
                    dfa[i] = 0
                    refs[0] += 1
            else:
                dfa[i] = 0
                refs[0] -= 1
                states_used -= 1
        else:
            return


def bfs_ordered(dfa, S=2):
    '''Return an equivalent DFA with states ordered by breadth-first search (and unreachable states stripped).'''
    n, used = len(dfa)//S, 1
    state_id = [None]*n
    state_id[0] = 0
    trans = []
    bfs_q = deque((0,))
    while bfs_q:
        for _ in range(S):
            trans.append(None)
        q0 = bfs_q.popleft()
        i0 = state_id[q0]
        for s in range(S):
            q1 = dfa[S*q0+s]
            i1 = state_id[q1]
            if i1 is None:
                i1 = state_id[q1] = used
                used += 1
                bfs_q.append(q1)
            trans[S*i0+s] = i1
    return trans


def reachable_states(dfa, S=2, up_to=0):
    '''Return the number of reachable states: like len(bfs_ordered(dfa, S))//S, but faster.'''
    reached = {0}
    visiting = [0]
    while visiting:
        q0 = visiting.pop()
        for s in range(S):
            q1 = dfa[S*q0+s]
            l0 = len(reached)
            reached.add(q1)
            if l0 != len(reached):
                visiting.append(q1)
            if len(reached) == up_to:
                return up_to
    return len(reached)


def product(dfas, S=2):
    '''Return a DFA with a state per possible tuple of states reached in the component DFAs.'''
    state_id = {(0,)*len(dfas): 0}
    trans = []
    bfs_q = deque(state_id)
    while bfs_q:
        for _ in range(S):
            trans.append(None)
        q0 = bfs_q.popleft()
        i0 = state_id[q0]
        for s in range(S):
            q1 = tuple([u[S*q+s] for (u, q) in zip(dfas, q0)])
            try:
                i1 = state_id[q1]
            except KeyError:
                i1 = state_id[q1] = len(state_id)
                bfs_q.append(q1)
            trans[S*i0+s] = i1
    return trans


def line_graph(dfa, S=2):
    '''Return a DFA whose states correspond to the possible transitions of the original. In other words, augment the DFA's state with the history going back one step.'''
    state_id = {(0, 0): 0}
    trans = []
    bfs_q = deque(state_id)
    while bfs_q:
        for _ in range(S):
            trans.append(None)
        q0, q1 = q0q1 = bfs_q.popleft()
        i0 = state_id[q0q1]
        for s in range(S):
            q2 = dfa[S*q1+s]
            q1q2 = q1, q2
            try:
                i1 = state_id[q1q2]
            except KeyError:
                i1 = state_id[q1q2] = len(state_id)
                bfs_q.append(q1q2)
            trans[S*i0+s] = i1
    return trans

def redirect(dfa, q_old, q_new):
    '''Return a DFA with transitions to "q_old" replaced with transitions to "q_new".'''
    return [(q_new if x == q_old else x) for x in dfa]
