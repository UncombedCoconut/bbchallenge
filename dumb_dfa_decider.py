#!/usr/bin/pypy3
# SPDX-FileCopyrightText: 2022 Justin Blanchard <UncombedCoconut@gmail.com>
# SPDX-License-Identifier: Apache-2.0 OR MIT
from bbchallenge import ithl, L
from dfa_utils import iter_dfas


def ctl_search(tm, nl, nr):
    N, S = tm.states, tm.symbols
    for l_dfa in iter_dfas(nl, S):
        for r_dfa in iter_dfas(nr, S):
            try:
                accept = [[[[False for qr in range(nr)] for s in range(S)] for state in range(N)] for ql in range(nl)]
                # Close the language by searching for states besides the initial which we must accept.
                def search(ql, f, r, qr):
                    if accept[ql][f][r][qr]:
                        return
                    accept[ql][f][r][qr] = True
                    w, d, t = tm.transition(f, r)
                    if t == -1:
                        raise InterruptedError  # Meaning "we hit a halt", LOL.
                    for s in range(S):
                        if d == L:
                            for l2 in range(nl):
                                if l_dfa[S*l2+s] == ql:
                                    search(l2, t, s, r_dfa[S*qr+w])
                        else: # d == R
                            for r2 in range(nr):
                                if r_dfa[S*r2+s] == qr:
                                    search(l_dfa[S*ql+w], t, s, r2)

                search(0, 0, 0, 0)
            except InterruptedError:
                continue
            return ctl_text(l_dfa, r_dfa, accept)


def ctl_text(l_dfa, r_dfa, accept):
    nl, N, S, nr = len(accept), len(accept[0]), len(accept[0][0]), len(accept[0][0][0])
    if args.re:
        from automata.fa import nfa, gnfa
        from itertools import product
        transitions = ({f'L{ql}': {str(s): set() for s in range(S)} for ql in range(nl)}
                     | {f'R{qr}': {str(s): set() for s in range(S)} for qr in range(nr)})
        for ql in range(nl):
            for s in range(S):
                transitions[f'L{ql}'][f'{s}'].add(f'L{l_dfa[S*ql+s]}')
        for qr in range(nr):
            for s in range(S):
                transitions[f'R{r_dfa[S*qr+s]}'][f'{s}'].add(f'R{qr}')
        # Give automata-lib single-character head symbols to work with.
        ord_to_head = {97 + S*f + r: f'({ithl(f)}{r})' for f in range(N) for r in range(S)}
        input_symbols = {str(s) for s in range(S)}.union(map(chr, ord_to_head))
        for (ql, f, r, qr) in product(range(nl), range(N), range(S), range(nr)):
            if accept[ql][f][r][qr]:
                head = chr(97 + S*f + r)
                transitions[f'L{ql}'].setdefault(head, set()).add(f'R{qr}')
        marvin = nfa.NFA(states=set(transitions), input_symbols=input_symbols, transitions=transitions, initial_state='L0', final_states={'R0'})
        return gnfa.GNFA.from_nfa(marvin).to_regex().translate(ord_to_head)
    else:
        return ' '.join([f'DFAs: {l_dfa=}, {r_dfa=}, accept:'] +
            [f'{ql}:{ithl(f)}@{s}:{qr}' for ql in range(nl) for f in range(N) for s in range(S) for qr in range(nr) if accept[ql][f][s][qr]])


if __name__ == '__main__':
    from bb_args import ArgumentParser, tm_args
    ap = ArgumentParser(description='If a Closed Tape Language of given complexity proves a TM cannot halt, show it.', parents=[tm_args()])
    ap.add_argument('-l', help='Max DFA states for left side', type=int, default=4)
    ap.add_argument('-r', help='Max DFA states for right side', type=int, default=4)
    ap.add_argument('--re', help='Output a regular expression (requires automata-lib)', action='store_false')
    args = ap.parse_args()

    for tm in args.machines:
        ctl = ctl_search(tm, args.l, args.r)
        if ctl:
            print(tm, 'infinite', ctl, sep=', ')
        else:
            print(tm, 'undecided', sep=', ')
