#!/usr/bin/pypy3
# SPDX-FileCopyrightText: 2022 Justin Blanchard <UncombedCoconut@gmail.com>
# SPDX-License-Identifier: Apache-2.0 OR MIT
from automata.fa import dfa, nfa
from collections import defaultdict, deque
RIP = {}

def to_dfas(path):
    exprs = []
    head_input_symbol = defaultdict(lambda: chr(97+len(head_input_symbol)))  # Give automata-lib arbitrary 1-char representations of, say, (B, 1).
    with open(path) as f:
        tm_text = next(f)
        for line in f:
            line = line.replace('(.)', '(0|1)').replace('@', '*')
            try:
                state_colon, left, read, right = line.split()
                read = int(read)
            except ValueError:
                continue
            head = head_input_symbol[state_colon.rstrip(':'), read]
            exprs.append(f'{left}{head}{right}')

    tape_alphabet = set(tm_text).intersection(map(str, range(10)))
    tm_symbols = int(max(tape_alphabet)) + 1
    expr = '|'.join(exprs)
    del exprs
    a = nfa.NFA.from_regex(expr, input_symbols=tape_alphabet.union(head_input_symbol.values()))
    alr = dfa.DFA.from_nfa(a).minify(), dfa.DFA.from_nfa(a.reverse()).minify()
    out = []

    for a in alr:
        # The pattern should start with 0*, and automata-lib should have to have a pattern that stabilizes after 0-transitions.
        q0 = a.initial_state
        for _ in a.states:
            q1 = a.transitions[q0]['0']
            if q0 == q1: break
            q0 = q1
        else:
            raise RuntimeError('Cannot find acceptable initial DFA state')

        state_id = {q0: 0}
        trans = []
        bfs_q = deque(state_id)
        while bfs_q:
            for _ in range(tm_symbols):
                trans.append(None)
            q0 = bfs_q.popleft()
            i0 = state_id[q0]
            for b in range(tm_symbols):
                q1 = a.transitions.get(q0, RIP).get(str(b))
                try:
                    i1 = state_id[q1]
                except KeyError:
                    i1 = state_id[q1] = len(state_id)
                    bfs_q.append(q1)
                trans[tm_symbols*i0+b] = i1
        out.append(trans)
    return out

if __name__ == '__main__':
    from argparse import ArgumentParser
    ap = ArgumentParser(description='Convert Franse Faase verification files to 1-side DFAs.')
    ap.add_argument('paths', help='path to the Frans Faase SymbolicTM file', nargs='+')
    args = ap.parse_args()
    for path in args.paths:
        print(*to_dfas(path))
