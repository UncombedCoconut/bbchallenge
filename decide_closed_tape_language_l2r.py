#!/usr/bin/pypy3
# SPDX-FileCopyrightText: 2022 Justin Blanchard <UncombedCoconut@gmail.com>
# SPDX-License-Identifier: MIT
from bbchallenge import DB_PATH, get_header, get_machine_i, ithl

def binary_DFAs(n):
    ''' We wish to solve a TM modulo an equivalence relation on the left-of-head tape configurations.
        This generator yields a list T representing every DFA (Q={0,…,n-1}, Σ={0,1}, δ(q,b)=T[2*q+b], q₀=0), such that:
        • T[0]=0 (invariance under leading 0's on the tape),
        • The list [T.index(q) for q in range(n)] is increasing (an arbitrary ordering of the states).'''
    dfa = [0] * (2*n)
    refs = [2] + [0]*(n-1)
    states_used = 1
    while True:
        if states_used == n:
            yield dfa
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

def quotient_PDS(tm, dfa):
    ''' A TM, modulo a DFA's classification of left-of-head configurations, becomes a "pushdown system" (P,Γ,Δ) under the conventions of
            Bouajjani, A., Esparza, J., & Maler, O. (1997, July). Reachability analysis of pushdown automata: Application to model-checking.
            In International Conference on Concurrency Theory (pp. 135-150). Springer, Berlin, Heidelberg.
            Available: https://www.irif.fr/~abou//BEM97.pdf
        where Γ={0,1} and P = {5*q+s+1: q∈Q a DFA state, s∈{0,…,4} a TM state} ∪ {HALT ≝ 0}.
        This function yields the transitions.
        CAUTION: in this formulation the stack (corresponding to the TM's right half-tape) is finite, not infinite and eventually zero-filled. '''
    for s in range(5):
        for b in range(2):
            write, move, goto = tm[6*s+3*b : 6*s+3*(b+1)]
            for q1b1, q2 in enumerate(dfa):
                q1, b1 = divmod(q1b1, 2)
                if not goto: # HALT rule for s@b - just need one PDS transition per DFA state.
                    if b1==0:
                        yield (5*q1+s+1, b), (0, ())
                elif move:  # LEFT rule: [δ(q1,b1)] s@b RHS => [q1] goto@b1 write RHS
                    yield (5*q2+s+1, b), (5*q1+goto, (b1, write))
                else:  # RIGHT rule: [q1] s@b RHS => [δ(q1,write)] goto@RHS
                    if b1 == write:
                        yield (5*q1+s+1, b), (5*q2+goto, ())

def right_half_tape_NFA(tm, dfa):
    ''' The same [BEM97] paper constructs a specialized NFA recognizing those PDS configurations from which (e.g.) a halting one is reachable.
        Technically, it's a "multi-automaton" for the PDS: instead of one initial state, it has states identified with the PDS's control states.
        To test a PDS configuration, start it at the control state and run it on the stack word.
        For present purposes, no auxiliary states are needed, so we simply reuse the PDS's state indices P, marking states F={HALT}={0} as final/accepting.
        This function returns a transition table of bitmasks: T such that (q,γ,q') is a transition iff T[2*q+γ] & (1<<q') != 0.
        CAUTION: we're still analyzing "configurations" where the stack / right half-tape is finite. '''
    nP = 5 * (len(dfa)//2) + 1
    transP = list(quotient_PDS(tm, dfa))
    # Start with a recognizer for HALT configurations. (The paper demands no transitions to initial states, but a loop at HALT doesn't affect its proof.)
    T = [1, 1] + [0]*(2*nP)
    # Add the transitions described in [BEM97] section 2.2, until none of them are new.
    grew = True
    while grew:
        grew = False
        for (j, b), (k, w) in transP:
            new_Tjb = T[2*j+b] | multi_step_NFA(T, k, w)
            if T[2*j+b] != new_Tjb:
                T[2*j+b] = new_Tjb
                grew = True
    return T

def step_NFA_mask(T, mask, bit):
    ''' Given NFA transition table T (as in right_half_tape_NFA), a bitmask of possible states, and a bit, return the bitmask of possible next states. '''
    new = 0
    for i, mi in enumerate(T[bit::2]):
        if mask & (1<<i):
            new |= mi
    return new

def multi_step_NFA(T, initial_state, bits):
    ''' Given NFA transition table T (as in right_half_tape_NFA), starting state, and a "bit" sequence in Γ*, return the bitmask of possible end states. '''
    mask = 1 << initial_state
    for bit in bits:
        mask = step_NFA_mask(T, mask, bit)
    return mask

def test_zero_stacks(T, initial_state=1):
    ''' Evaluate whether the NFA accepts *any* configuration with the given PDS control state and a stack of arbitrarily many zeros. '''
    old = 0
    new = 1 << initial_state
    while old != new:
        old, new = new, new | step_NFA_mask(T, new, 0)
    return bool(new & 1)

def ctl_search(tm, l_states_max):
    ''' Return a Closed Tape Language which recognizes all halting configurations of the TM but not the intial state... if we find one. '''
    for l_states in range(1, l_states_max+1):
        for l_dfa in binary_DFAs(l_states):
            r_nfa = right_half_tape_NFA(tm, l_dfa)
            if not test_zero_stacks(r_nfa): # Even modulo l_dfa's equivalence on left half-tapes, halting states are unreachable from [0] A@00....0.
                return CTL(l_dfa, r_nfa)
    return False


class CTL:
    ''' A displayable Closed Tape Language. For purposes of this decider, a tape's format is [01]*[A-E][01]*, with the head [A-E] on the following digit. '''
    def __init__(self, l_dfa, r_nfa):
        self.l_dfa, self.r_nfa = l_dfa, r_nfa

    def __str__(self):
        ''' Return a regexp representation. Requires automata-lib.
            This code is allowed to be icky because the point is to output an untrusted, verifiable certificate of the result. '''
        from automata.fa import nfa, gnfa
        import re

        nL, nR = len(self.l_dfa)//2, len(self.r_nfa)//2
        # Define labels for the left, right automata states. This is our chance to eliminate redundant states of r_nfa.
        # (Any TM tape may be taken to end with an arbitrary string of zeros, so it's safe to identify states with identical 0/1 transitions.)
        QL = 'L{}'.format
        QR = lambda r: f'R{r_state_map[r]}'
        r_trans = list(zip(self.r_nfa[0::2], self.r_nfa[1::2]))
        r_state_map = [r_trans.index(trans) for trans in r_trans]

        transitions = {QL(l): {} for l in range(nL)} | {QR(r): {} for r in range(nR)}
        for l1b, l2 in enumerate(self.l_dfa):
            l1, b = divmod(l1b, 2)
            transitions[QL(l1)][f'{b}'] = {QL(l2)}
        for r1b, rmask in enumerate(self.r_nfa):
            r1, b = divmod(r1b, 2)
            transitions[QR(r1)][f'{b}'] = {QR(r2) for r2 in range(nR) if rmask&(1<<r2)}
        for l in range(nL):
            for s in range(5):
                transitions[QL(l)][ithl(s)] = {QR(5*l+s+1)}
        final_states = {QR(r) for r in range(nR) if test_zero_stacks(self.r_nfa, r)}
        marvin = nfa.NFA(states=set(transitions), input_symbols=set('01ABCDE'), transitions=transitions, initial_state='L0', final_states=final_states)
        expr = gnfa.GNFA.from_nfa(marvin).to_regex()

        # We're theoretically done, but automata-lib fails really hard at simplifying RE's.
        # Collapse idiocy like ((A|B)|C) to equivalents like (A|B|C).
        for _ in range(4):
            expr = re.sub(r'([|(])\(([^()+?*]+)\)([|)])', r'\1\2\3', expr)
        # Use character-class syntax where applicable.
        for i in range(6, 1, -1):
            expr = re.sub(r'\(' + r'\|'.join(['(.)']*i) + r'\)',  '[' + ''.join([fr'\{j+1}' for j in range(i)]) + ']' , expr)
            expr = re.sub(r'\|' + r'\|'.join(['(.)']*i) + r'\)', '|[' + ''.join([fr'\{j+1}' for j in range(i)]) + ']' , expr)
            expr = re.sub(r'\(' + r'\|'.join(['(.)']*i) + r'\|',  '[' + ''.join([fr'\{j+1}' for j in range(i)]) + ']|', expr)
        return expr


if __name__ == '__main__':
    from argparse import ArgumentParser
    ap = ArgumentParser(description='If a Closed Tape Language of given complexity proves a TM cannot halt, show it.')
    ap.add_argument('--db', help='Path to DB file', type=str, default=DB_PATH)
    ap.add_argument('-l', help='State limit for the left half-tape DFA. At worst, 5 (default) should be sub-second and 6 sub-minute.', type=int, default=5)
    ap.add_argument('-q', '--quiet', help='Do not output regexp proofs (for speed or to avoid depending on automata-lib)', action='store_true')
    ap.add_argument('seeds', help='DB seed numbers', type=int, nargs='*', default=[])
    args = ap.parse_args()

    for seed in args.seeds or range(int.from_bytes(get_header(args.db)[8:12], byteorder='big')):
        tm = get_machine_i(args.db, seed)
        ctl = ctl_search(tm, args.l)
        if ctl and args.quiet:
            print(seed, 'infinite', sep=', ')
        elif ctl:
            print(seed, 'infinite', ctl_search(tm, args.l), sep=', ')
        else:
            print(seed, 'undecided')
