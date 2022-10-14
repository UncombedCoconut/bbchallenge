#!/usr/bin/pypy3
# SPDX-FileCopyrightText: 2022 Justin Blanchard <UncombedCoconut@gmail.com>
# SPDX-License-Identifier: Apache-2.0 OR MIT
from bbchallenge import get_header, get_machine_i
from string_rewrite import Word, Rewrite, RewriteSystem
from decide_closed_tape_language_l2r import binary_DFAs, step_NFA_mask, multi_step_NFA, test_zero_stacks

def multi_step_DFA(T, state, srs_word_side):
    bits = map(int, srs_word_side)
    for bit in bits:
        state = T[2*state+bit]
    return state

def right_half_tape_NFA(srs, l_dfa):
    nL = len(l_dfa) // 2
    # NFA states: 0=halt, nL*(input_state) + dfa_state + 1.

    # Let's re-present the parts of the system that can lead to a halt, as before/after pairs of (NFA state ID | word).
    # Since we can't do ε-transitions, ensure any "before" word has at least one bit on the right.
    srs_trans = []
    for rw in srs.rewrites:
        if rw.f.r:
            srs_trans.append((rw.f, rw.t))
        else:
            srs_trans.extend(((rw.f+'0', rw.t+'0'), (rw.f+'1', rw.t+'1')))
    for hw in srs.special_words['halting']:
        if hw.r:
            srs_trans.append((hw, 0))
        else:
            srs_trans.extend(((hw+'0', 0), (hw+'1', 0)))
    # Enumerate the states where TM/~ needs to consume a bit.
    near_matches = sorted({w.s + w.r[:-i] for (w, _) in srs_trans for i in range(len(w.r))})
    # Construct the transitions between these states, and record the state IDs for each symbol.
    r_nfa = [1, 1] + [0]*(2*nL*len(near_matches))
    glue = {}
    for i, nm in enumerate(near_matches):
        if len(nm) == 1:
            glue[nm] = i
        else:
            i_prefix, suffix = near_matches.index(nm[:-1]), int(nm[-1])
            for q in range(nL):
                r_nfa[2*(nL*i_prefix+q+1) + suffix] = 1 << (nL*i+q+1)

    # Represent each transition as a pair of NFA paths, represented as (start state, tuple_of_bit_values).
    transP = [
        [
            (i_w, ()) if type(i_w) is int
            else (
                nL * glue[i_w.s] + multi_step_DFA(l_dfa, q, i_w.l) + 1,
                tuple(map(int, i_w.r))
            )
            for i_w in trans
        ]
        for q in range(nL) for trans in srs_trans
    ]
    # On the "read" side, transitions must be allowed to proceed with as little *new* input as possible.
    for jr_kw in transP:
        (j, r) = jr_kw[0]
        while len(r) > 1 and (mask := r_nfa[2*j+r[0]]):
            j, r = mask.bit_length()-1, r[1:]
        jr_kw[0] = (j, r)

    # TODO: Optimization: Pre-apply anything in transP where len(r)==1 and len(w)==0.

    grew = True
    while grew:
        grew = False
        for (j, r), (k, w) in transP:
            from_mask = multi_step_NFA(r_nfa, j, r[:-1])
            r_end = r[-1]
            to_mask = multi_step_NFA(r_nfa, k, w)
            for j_end, old_Tjb in enumerate(r_nfa[r_end::2]):
                if from_mask & (1 << j_end):
                    new_Tjb = old_Tjb | to_mask
                    if old_Tjb != new_Tjb:
                        r_nfa[2*j_end + r_end] = new_Tjb
                        grew = True
    return r_nfa, glue

def ctl_search(srs, l_states_max):
    ''' Return a Closed Tape Language which recognizes all halting configurations of the TM but not the intial state... if we find one. '''
    mirror_srs = srs.mirror()

    for l_states in range(1, l_states_max+1):
        for mirrored in False, True:
            for l_dfa in binary_DFAs(l_states):
                working_srs = mirror_srs if mirrored else srs
                r_nfa, glue = right_half_tape_NFA(working_srs, l_dfa)
                nL, nR = len(l_dfa)//2, len(r_nfa)//2
                start = multi_step_DFA(l_dfa, 0, working_srs.start.l)
                start = multi_step_NFA(r_nfa, nL * glue[working_srs.start.s] + start + 1, map(int, working_srs.start.r))
                if not any(test_zero_stacks(r_nfa, k) for k in range(nR) if start & (1<<k)):
                    return CTL(l_dfa, glue, r_nfa, mirrored)
    return False


class CTL:
    ''' A displayable Closed Tape Language. '''
    def __init__(self, l_dfa, glue, r_nfa, mirrored=False):
        self.l_dfa, self.glue, self.r_nfa, self.mirrored = l_dfa, glue, r_nfa, mirrored

    def __str__(self):
        ''' Return a regexp representation. Requires automata-lib.
            This code is allowed to be icky because the point is to output an untrusted, verifiable certificate of the result. '''
        from automata.fa import dfa, nfa, gnfa
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
            for sym, s in self.glue.items():
                transitions[QL(l)][sym] = {QR(nL*s+l+1)}
        final_states = {QR(r) for r in range(nR) if test_zero_stacks(self.r_nfa, r)}

        # Inspired by Brzozowski's simple algorithm for *D*FA minimization, we try two automata: one constructed simply, and one determinized (which surprisingly often helps instead of hurting).
        marvin = nfa.NFA(states=set(transitions), input_symbols={'0', '1'}.union(self.glue), transitions=transitions, initial_state='L0', final_states=final_states)
        if self.mirrored:
            marvin = marvin.reverse()
            bender = nfa.NFA.from_dfa(dfa.DFA.from_nfa(marvin).minify())
        else:
            bender = nfa.NFA.from_dfa(dfa.DFA.from_nfa(marvin.reverse()).minify()).reverse()
        expr = min(gnfa.GNFA.from_nfa(marvin).to_regex() or '$^', gnfa.GNFA.from_nfa(bender).to_regex() or '$^', key=len)

        # We're theoretically done, but automata-lib fails really hard at simplifying RE's.
        # Collapse idiocy like ((A|B)|C) to equivalents like (A|B|C).
        for _ in range(4):
            expr = re.sub(r'([|(])\(([^()+?*]+)\)([|)])', r'\1\2\3', expr)
        # Use character-class syntax where applicable.
        for i in range(6, 1, -1):
            expr = re.sub(r'\(' + r'\|'.join([ '(.)' ]*i) + r'\)',   '[' + ''.join([fr'\{j+1}' for j in range(i)]) + ']'  , expr)
            expr = re.sub(r'\|' + r'\|'.join([ '(.)' ]*i) + r'\)',  '|[' + ''.join([fr'\{j+1}' for j in range(i)]) + '])' , expr)
            expr = re.sub(r'\(' + r'\|'.join([ '(.)' ]*i) + r'\|',  '([' + ''.join([fr'\{j+1}' for j in range(i)]) + ']|' , expr)
        return expr


if __name__ == '__main__':
    from argparse import ArgumentParser
    ap = ArgumentParser(description='If a Closed Tape Language of given complexity proves a TM cannot halt, show it.')
    ap.add_argument('--db', help='Path to DB file', type=str, default='all_5_states_undecided_machines_with_global_header')
    ap.add_argument('-l', help='State limit for the left half-tape DFA. At worst, 5 (default) should be sub-second and 6 sub-minute.', type=int, default=5)
    ap.add_argument('-q', '--quiet', help='Do not output regexp proofs (for speed or to avoid depending on automata-lib)', action='store_true')
    ap.add_argument('seeds', help='DB seed numbers', type=int, nargs='*', default=[])
    args = ap.parse_args()

    for seed in args.seeds or range(int.from_bytes(get_header(args.db)[8:12], byteorder='big')):
        tm = get_machine_i(args.db, seed)
        srs = RewriteSystem(tm)
        srs.simplify()
        if not srs.rewrites: # Accidentally solved already?
            print(seed, 'halts' if srs.starts_in_state('halting') else 'infinite, cycler', sep=', ')
            continue
        ctl = ctl_search(srs, args.l)
        if ctl and not args.quiet:
            reason = dict(coctl=str(ctl), start=str(srs.start), rewritten='|'.join(str(rw.f) for rw in srs.rewrites)) | {k: '|'.join(map(str, v)) for k,v in srs.special_words.items() if v}
            print(seed, 'infinite', reason, sep=', ')
        else:
            print(seed, 'infinite' if ctl else 'undecided', sep=', ')
