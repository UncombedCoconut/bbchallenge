# SPDX-FileCopyrightText: 2023 Justin Blanchard <UncombedCoconut@gmail.com>
# SPDX-License-Identifier: Apache-2.0 OR MIT


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
