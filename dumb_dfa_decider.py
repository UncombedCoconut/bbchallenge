#!/usr/bin/pypy3
# SPDX-FileCopyrightText: 2022 Justin Blanchard <UncombedCoconut@gmail.com>
# SPDX-License-Identifier: MIT
from bbchallenge import DB_PATH, get_machine_i, ithl


def bin_dfa_iter(n):
    ''' Models a deterministic finite automaton on the alphabet [01]. States are range(n). Initial state is 0.
        Representation is a sequence of the 2n destinations: new_state = dfa[2*state+bit].
        This is for TM tapes, starting at the outer edge, so we force a transition 0 -0-> 0.
        The full tape-language DFA is to be formed by gluing two of these together at the middle:
        accept or not based on (left_state, head_state, head_bit, right_state). '''
    dfa = [0] * (2*n)
    refs = [2] + [0]*(n-1)
    states_used = 1
    while True:
        yield dfa, states_used
        for i in reversed(range(1, 2*states_used)):
            refs[dfa[i]] -= 1
            if refs[dfa[i]]:
                if dfa[i] < n-1:
                    dfa[i] += 1
                    refs[dfa[i]] += 1
                    if dfa[i] == states_used:
                        states_used += 1
                        refs[0] += 2
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


def ctl_search(machine, nLmax, nRmax):
    for L, nL in bin_dfa_iter(nLmax):
        for R, nR in bin_dfa_iter(nRmax):
            try:
                accept = [[[[False for r in range(nR)] for bit in range(2)] for state in range(5)] for l in range(nL)]
                # Close the language by searching for states besides the initial which we must accept.
                def search(l, s, b, r):
                    if accept[l][s][b][r]:
                        return
                    accept[l][s][b][r] = True
                    write = machine[6*s+3*b]
                    move = machine[6*s+3*b+1]
                    goto = machine[6*s+3*b+2]-1
                    if goto == -1:
                        raise InterruptedError  # Meaning "we hit a halt", LOL.
                    # Move (LEFT if move else RIGHT)
                    for b2 in range(2):
                        if move:  # LEFT
                            for l2 in range(nL):
                                if L[2*l2+b2] == l:
                                    search(l2, goto, b2, R[2*r+write])
                        else: # RIGHT
                            for r2 in range(nR):
                                if R[2*r2+b2] == r:
                                    search(L[2*l+write], goto, b2, r2)

                search(0, 0, 0, 0)
            except InterruptedError:
                continue
            print(f'DFAs: {L=}, {R=}, accept:', *[f'{l}:{ithl(s)}@{b}:{r}' for l in range(nL) for s in range(5) for b in range(2) for r in range(nR) if accept[l][s][b][r]])
            return


if __name__ == '__main__':
    from argparse import ArgumentParser
    ap = ArgumentParser(description='If a Closed Tape Language of given complexity proves a TM cannot halt, show it.')
    ap.add_argument('--db', help='Path to DB file', type=str, default=DB_PATH)
    ap.add_argument('-l', help='Max DFA states for left side', type=int, default=4)
    ap.add_argument('-r', help='Max DFA states for right side', type=int, default=4)
    ap.add_argument('seeds', help='DB seed numbers', type=int, nargs='*', default=[7410754])
    args = ap.parse_args()

    for seed in args.seeds:
        print('='*40, seed, '='*40)
        machine = get_machine_i(args.db, seed)
        ctl_search(machine, args.l, args.r)
