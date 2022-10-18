#!/usr/bin/pypy3
# SPDX-FileCopyrightText: 2022 Justin Blanchard <UncombedCoconut@gmail.com>
# SPDX-License-Identifier: Apache-2.0 OR MIT
from automata.fa import dfa, gnfa, nfa
from bbchallenge import get_header, get_machine_i, ithl, L, R
from enum import IntEnum
from itertools import chain, product
import logging
import re
from z3 import And, Bool, Implies, Not, Or, PbEq, Solver, sat

ExactlyOne = lambda conds: PbEq([(cond, 1) for cond in conds], 1)

# Special state IDs
DFA_INIT = TM_INIT = 0
NFA_HALT = 'Z'

class Mode(IntEnum):
    decision, model, nfa, re = range(4)

class Search:
    def __init__(self, n, tm_states=5):
        self.n = n
        # Goal: a finite state machine that accepts all halting configurations (at least), scanning deterministically then non. Define the state spaces:
        self.Q_tm = range(tm_states)
        self.Q_dfa = range(self.n)
        self.Q_nfa = {NFA_HALT}.union(product(self.Q_dfa, self.Q_tm))
        # The combined output will have...
        self.nfa_args = dict(states=self.Q_nfa.union(self.Q_dfa), input_symbols=set('01').union(map(ithl, self.Q_tm)), initial_state=DFA_INIT)
        # We use a SAT solver with the following variables representing the problem. (They have accessor methods below.)
        # Also, convention for TM rules: From state F, if bit r read, write bit w, move direction lr, and go to state T.
        self.z3 = Solver()
        self._tm_write = {(F, r): Bool(f'TM_{F}{r}_write') for (F, r) in product(self.Q_tm, range(2))}
        self._tm_right = {(F, r): Bool(f'TM_{F}{r}_right') for (F, r) in product(self.Q_tm, range(2))}
        self._tm_halts = {(F, r): Bool(f'TM_{F}{r}_halts') for (F, r) in product(self.Q_tm, range(2))}
        self._tm_to = {(F, r, T): Bool(f'TM_{F}{r}_to_{T}') for (F, r, T) in product(self.Q_tm, range(2), self.Q_tm)}
        self._dfa = {(q1, r, q2): Bool(f'dfa_{q1}_{r}_{q2}') for (q1, r, q2) in product(self.Q_dfa, range(2), self.Q_dfa)}
        self._nfa = {(x1, r, x2): Bool(f'nfa_{x1}_{r}_{x2}') for (x1, r, x2) in product(self.Q_nfa, range(2), self.Q_nfa)}
        self._accept = {x: Bool(f'accept_{x}') for x in self.Q_nfa}
        # Set up most of the formulae ahead of time.
        self.z3.add(list(chain(self.dfa_rules(), self.nfa_rules(), self.tm_rules(), self.closure_rules())))
        logging.getLogger('\U0001f9ab'*3).info('Ready to solve for DFA size %s', n)

    def dfa(self, q1, r, q2): return self._dfa[q1, r, q2]
    def nfa(self, x1, r, x2): return self._nfa[x1, r, x2]
    def accept(self, x): return self._accept[x]
    def tm_write(self, F, r, w): return self._tm_write[F, r] == bool(w)
    def tm_moves(self, F, r, lr): return self._tm_right[F, r] == (lr==R)
    def tm_halts(self, F, r): return self._tm_halts[F, r]
    def tm_to(self, F, r, T): return self._tm_to[F, r, T]

    def tm_rules(self):
        for F, r in product(self.Q_tm, range(2)):
            # Exactly one "to" state per TM transition.
            yield ExactlyOne([self.tm_to(F, r, T) for T in self.Q_tm] + [self.tm_halts(F, r)])

    def machine_code(self, tm):
        for Fr, (w, lr, T_incr) in enumerate(zip(tm[::3], tm[1::3], tm[2::3])):
            F, r, T = Fr//2, Fr%2, T_incr-1
            if T < 0:
                yield self.tm_halts(F, r)
            else:
                yield And(self.tm_write(F, r, w), self.tm_moves(F, r, lr), self.tm_to(F, r, T))

    def dfa_rules(self):
        # Determinism: exactly one transition per (state, bit_read)
        for q1, r in product(self.Q_dfa, range(2)):
            yield ExactlyOne([self.dfa(q1, r, q2) for q2 in self.Q_dfa])

        # Insensitivity to leading zeros
        yield self.dfa(DFA_INIT, 0, DFA_INIT)

        # Symmetry breaking: if we tabulate [δ(q, r) for q in range(self.n) for r in range(2)], the vertices are seen in order.
        seen = {(q, t): Bool(f'{q} in dfa_table[:{t+1}]') for q in self.Q_dfa for t in range(2*self.n+1)}
        for q in self.Q_dfa:
            yield seen[q, 0] == (q==0)
        for q, t in product(range(self.n-1), range(2*self.n+1)):
            yield Implies(seen[q+1, t], seen[q, t])
        for q, t in product(self.Q_dfa, range(2*self.n)):
            yield seen[q, t+1] == Or(seen[q, t], And(seen[t//2, t], self.dfa(t//2, t%2, q)))

    def nfa_rules(self):
        # Once halted, always halted.
        yield self.nfa(NFA_HALT, 0, NFA_HALT)
        yield self.nfa(NFA_HALT, 1, NFA_HALT)
        # Trailing zeros are not necessary.
        for x, y in product(self.Q_nfa, repeat=2):
            yield Implies(And(self.nfa(x, 0, y), self.accept(y)), self.accept(x))
        # The machine separates the initial configuration from a halted one (and, with the closure rules in place, any potentially halting one).
        yield Not(self.accept((DFA_INIT, TM_INIT)))
        yield self.accept(NFA_HALT)

    def closure_rules(self):
        # If we can reach the configuration after a transition (q1, F) right_tape -> (q2, T) right_tape', we can reach the configuration before.
        for F, r in product(self.Q_tm, range(2)):
            halt_rule = self.tm_halts(F, r)
            for q in self.Q_dfa:
                yield Implies(halt_rule, self.nfa((q, F), r, NFA_HALT))
            for w, T in product(range(2), self.Q_tm):
                this = And(self.tm_write(F, r, w), self.tm_to(F, r, T))
                left_rule = And(this, self.tm_moves(F, r, L))
                right_rule = And(this, self.tm_moves(F, r, R))
                for q1, q2 in product(self.Q_dfa, self.Q_dfa):
                    yield Implies(And(right_rule, self.dfa(q1, w, q2)), self.nfa((q1, F), r, (q2, T)))
                    for b1, x1, x2 in product(range(2), self.Q_nfa, self.Q_nfa):
                        yield Implies(And(left_rule, self.dfa(q1, b1, q2), self.nfa((q1, T), b1, x1), self.nfa(x1, w, x2)), self.nfa((q2, F), r, x2))

    def __call__(self, tm, mode, mirrored=False):
        if mirrored:
            tm = left_right_reversal(tm)
        solution = (self.z3.check(*self.machine_code(tm)) == sat)
        if mode >= Mode.model and solution:
            solution = self.z3.model()
        if mode >= Mode.nfa and solution:
            transitions  = {q0: {str(r): {q1 for q1 in self.Q_dfa if solution[self.dfa(q0, r, q1)]} for r in range(2)} for q0 in self.Q_dfa}
            transitions |= {x0: {str(r): {x1 for x1 in self.Q_nfa if solution[self.nfa(x0, r, x1)]} for r in range(2)} for x0 in self.Q_nfa}
            for q, S in product(self.Q_dfa, self.Q_tm):
                transitions[q][ithl(S)] = {(q, S)}
            final_states = {x for x in self.Q_nfa if solution[self.accept(x)]}
            solution = dict(transitions=transitions, final_states=final_states, **self.nfa_args)
        if mode >= Mode.re and solution:
            solution = nfa.NFA(**solution)
            # Inspired by Brzozowski's simple algorithm for *D*FA minimization, we try two automata: one constructed simply,
            # and one determinized (which surprisingly often helps instead of hurting).
            if mirrored:
                solution = solution.reverse()
                solution_RDR = nfa.NFA.from_dfa(dfa.DFA.from_nfa(solution).minify())
            else:
                solution_RDR = nfa.NFA.from_dfa(dfa.DFA.from_nfa(solution.reverse()).minify()).reverse()
            solution = min(gnfa.GNFA.from_nfa(solution).to_regex(), gnfa.GNFA.from_nfa(solution_RDR).to_regex(), key=len)
            # Indicate which adjacent bit corresponds to the head's state.
            solution = re.sub(r'([A-Z])', r'@\1' if mirrored else r'\1@', solution)
            # Desperate attempts to simplify REs, as in other scripts.
            for _ in range(4):
                solution = re.sub(r'([|(])\(([^()+?*]+)\)([|)])', r'\1\2\3', solution)
            # Use character-class syntax where applicable.
            for i in range(6, 1, -1):
                solution = re.sub(r'\(' + r'\|'.join([ '(.)' ]*i) + r'\)',   '[' + ''.join([fr'\{j+1}' for j in range(i)]) + ']'  , solution)
                solution = re.sub(r'\|' + r'\|'.join([ '(.)' ]*i) + r'\)',  '|[' + ''.join([fr'\{j+1}' for j in range(i)]) + '])' , solution)
                solution = re.sub(r'\(' + r'\|'.join([ '(.)' ]*i) + r'\|',  '([' + ''.join([fr'\{j+1}' for j in range(i)]) + ']|' , solution)
                solution = re.sub(r'\(' + r'\|'.join([ '(.)@']*i) + r'\)',   '[' + ''.join([fr'\{j+1}' for j in range(i)]) + ']@' , solution)
                solution = re.sub(r'\|' + r'\|'.join([ '(.)@']*i) + r'\)',  '|[' + ''.join([fr'\{j+1}' for j in range(i)]) + ']@)', solution)
                solution = re.sub(r'\(' + r'\|'.join([ '(.)@']*i) + r'\|',  '([' + ''.join([fr'\{j+1}' for j in range(i)]) + ']@|', solution)
        return solution

def left_right_reversal(tm):
    mirror_tm = bytearray(tm)
    for i in range(0, len(tm), 3):
        mirror_tm[i+1] ^= 1
    return bytes(mirror_tm)

if __name__ == '__main__':
    from argparse import ArgumentParser
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s: %(message)s')
    ap = ArgumentParser(description='Try to prove the supplied Turing Machines do not halt.')
    ap.add_argument('-d', '--db', help='Path to DB file', default='all_5_states_undecided_machines_with_global_header')
    ap.add_argument('-l', help='Size limit for the state machine (NFA) that should gnaw on the tape.', type=int, default=5)
    ap.add_argument('-x', help='Exclude DFAs this small. (Assume we tried already.).', type=int, default=0)
    ap.add_argument('-m', '--mode', help='Level of detail to output.', choices=[m.name for m in Mode], default='re')
    ap.add_argument('seeds', help='DB seed numbers', type=int, nargs='*', default=[])
    args = ap.parse_args()

    searches = [Search(n) for n in reversed(range(args.l, args.x, -1))]
    for seed in args.seeds or range(int.from_bytes(get_header(args.db)[8:12], byteorder='big')):
        tm = get_machine_i(args.db, seed)
        attempts = (search(tm, Mode[args.mode], mirrored=mirrored) for search in searches for mirrored in (False, True))
        solution = next(filter(None, attempts), None)
        if solution:
            print(seed, 'infinite', solution, sep=', ')
        else:
            print(seed, 'undecided', sep=', ')
