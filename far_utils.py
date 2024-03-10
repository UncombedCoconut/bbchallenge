#!/usr/bin/pypy3
# SPDX-FileCopyrightText: 2023 Justin Blanchard <UncombedCoconut@gmail.com>
# SPDX-License-Identifier: Apache-2.0 OR MIT
from collections import deque
from itertools import permutations
from finite_automata_reduction import right_half_tape_NFA, test_solution, test_zero_stacks
from dfa_utils import bfs_ordered, reachable_states, redirect

def optimize_proof(tm, dfa, side=0, sim_space=16, sim_time=2**15, verbose=True):
    S = tm.symbols
    Q = len(dfa)//S
    tm = reversed(tm) if side else tm
    sink = next((q for q in range(Q) if set(dfa[S*q:S*(q+1)]) == {q}), None)
    sort_key = lambda q1q2: (reachable_states(redirect(dfa, *q1q2), S), q1q2[1], q1q2[0])

    non_sinking = [set() for _ in range(Q)]
    if sink is not None:
        code = tm.code
        qs = [0] * (sim_space+1)
        fr, l, r = 0, 0, 0
        for _ in range(sim_time):
            w, d, t = code[3*fr:3*fr+3]
            if d:
                fr, l, r = l%S + (t-1)*S, l//S, w + r*S
                if len(qs) > (sim_space+1):
                    qs.pop()
            else:
                fr, l, r = r%S + (t-1)*S, w + l*S, r//S
                qs.append(dfa[S*qs[-1]+w])
            history = ()
            shift = l
            for x in range(sim_space):
                history = (shift % S,) + history
                non_sinking[qs[-2-x]].add(history)
                shift //= S

    def after(q, word):
        for s in word:
            q = dfa[S*q+s]
        return q

    identifications = [(q1, q2) for (q1, q2) in permutations(range(Q), 2) if all(after(q2, w)!=sink for w in non_sinking[q1])]
    identifications.sort(key=sort_key)
    dfa_reduced = dfa

    if verbose:
        print('LR'[side], end='')
    for iq, (q1, q2) in enumerate(identifications):
        q_unreduced = redirect(dfa, q1, q2)
        q_reduced = bfs_ordered(q_unreduced, S)
        if len(q_reduced) >= len(dfa_reduced): continue
        status = f'?{len(q_reduced)//S}|{(1+iq)*100/len(identifications):.3f}%'
        if verbose:
            print(status, end='', flush=True)
            print('\b \b'*len(status), end='', flush=False)
        if test_solution(tm, q_reduced):
            dfa = q_unreduced
            dfa_reduced = q_reduced
            identifications[iq+1:] = sorted(identifications[iq+1:], key=sort_key)
            if verbose:
                print(f'>{len(q_reduced)//S}', end='', flush=True)
    if verbose:
        print(f' => {len(dfa_reduced)//S}', flush=True)
    return dfa_reduced

def complementary_dfa(tm, dfa, side):
    tm = reversed(tm) if side else tm
    T = right_half_tape_NFA(tm, dfa)
    Q = len(T[0])
    S = tm.symbols
    a = sum(1<<q for q in range(Q) if test_zero_stacks(T, q))
    state_id = {a: 0}
    trans = []
    bfs_q = deque(state_id)
    while bfs_q:
        for _ in range(S):
            trans.append(None)
        q0 = bfs_q.popleft()
        i0 = state_id[q0]
        for s in range(S):
            q1 = sum(1<<q for q in range(Q) if T[s][q] & q0)
            try:
                i1 = state_id[q1]
            except KeyError:
                i1 = state_id[q1] = len(state_id)
                bfs_q.append(q1)
            trans[S*i0+s] = i1
    return trans

def optimize_dfa_pair(tm, dfas, sim_space=16, sim_time=2**15, complement=False):
    new_dfa = dfas
    best_dfa = [None, None]
    while any(new_dfa):
        cur_dfa, new_dfa = new_dfa, [None, None]
        for side, dfa in enumerate(cur_dfa):
            if dfa:
                dfa = optimize_proof(tm, dfa, side, sim_space=sim_space, sim_time=sim_time)
                if not best_dfa[side] or len(dfa) < len(best_dfa[side]):
                    new_dfa[side] = best_dfa[side] = dfa
                    if complement:
                        new_dfa[1 - side] = complementary_dfa(tm, dfa, side)
    return best_dfa

if __name__ == '__main__':
    from bb_args import ArgumentParser, tm_args
    ap = ArgumentParser(description='optimize a DFA proof', parents=[tm_args()])
    ap.add_argument('-l', '--left', help='try to beat this left-tape DFA', type=int, nargs='*')
    ap.add_argument('-r', '--right', help='try to beat this right-tape DFA', type=int, nargs='*')
    ap.add_argument('-c', '--complement', help='using a 1-sided DFA, find one on the opposite side', action='store_true')
    ap.add_argument('-x', '--sim-space', help='do a simulation, tracking this much tape, to constrain attempted optimizations', type=int, default=16)
    ap.add_argument('-t', '--sim-time', help='run the constraining simulation for this amount of time', type=int, default=2**15)
    args = ap.parse_args()
    for tm in args.machines:
        best_dfa = optimize_dfa_pair(tm, [args.left, args.right], sim_space=args.sim_space, sim_time=args.sim_time, complement=args.complement)
        for side, dfa in enumerate(best_dfa):
            print(tm, 'LR'[side], dfa)
