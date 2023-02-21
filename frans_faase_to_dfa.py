#!/usr/bin/pypy3
# SPDX-FileCopyrightText: 2022 Justin Blanchard <UncombedCoconut@gmail.com>
# SPDX-License-Identifier: Apache-2.0 OR MIT
from automata.fa import dfa, nfa, gnfa
from collections import deque
from contextlib import contextmanager, nullcontext
from sys import stderr
from time import perf_counter

@contextmanager
def Timer(name):
    t0 = perf_counter()
    yield
    print(f'{name}:', perf_counter()-t0, file=stderr, flush=True)

def exprs(path):
    with open(path) as f:
        for line in f:
            line = line.replace('(.)', '(0|1)').replace('@', '*')
            try:
                state_colon, left, read, right = line.split()
                read = int(read)
            except ValueError:
                continue
            head = (str.upper if read else str.lower)(state_colon.rstrip(':'))
            yield f'{left}{head}{right}'

def to_dfa(path, right=False, verbose=True):
    with Timer('build regex') if verbose else nullcontext():
        expr = '|'.join(exprs(path))
    with Timer('build NFA') if verbose else nullcontext():
        a = nfa.NFA.from_regex(expr)
    with Timer('determinize') if verbose else nullcontext():
        a = dfa.DFA.from_nfa(a.reverse() if right else a)
    with Timer('minify') if verbose else nullcontext():
        a = a.minify()

    # The pattern should start with 0*, and automata-lib should have to have a pattern that stabilizes after 0-transitions.
    q0 = a.initial_state
    for _ in a.states:
        q1 = a.transitions[q0]['0']
        if q0 == q1: break
        q0 = q1
    else:
        raise RuntimeError('Cannot find acceptable initial DFA state')

    state_id = {a.initial_state: 0}
    trans = []
    bfs_q = deque(state_id)
    with Timer('BFS') if verbose else nullcontext():
        while bfs_q:
            trans.extend((None, None))
            q0 = bfs_q.popleft()
            i0 = state_id[q0]
            for b in range(2):
                q1 = a.transitions[q0][str(b)]
                try:
                    i1 = state_id[q1]
                except KeyError:
                    i1 = state_id[q1] = len(state_id)
                    bfs_q.append(q1)
                trans[2*i0+b] = i1
    return trans

if __name__ == '__main__':
    from argparse import ArgumentParser
    ap = ArgumentParser(description='Convert Franse Faase verification files to 1-side DFAs.')
    ap.add_argument('-r', '--right', help='translate the right side', action='store_true')
    ap.add_argument('paths', help='path to the Frans Faase SymbolicTM file', nargs='+')
    args = ap.parse_args()
    for path in args.paths:
        print(to_dfa(path, args.right))
