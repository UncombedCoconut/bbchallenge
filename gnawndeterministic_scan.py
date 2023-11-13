#!/usr/bin/pypy3
# SPDX-FileCopyrightText: 2022 Justin Blanchard <UncombedCoconut@gmail.com>
# SPDX-License-Identifier: Apache-2.0 OR MIT
from automata.fa import dfa, gnfa, nfa
from bbchallenge import ithl, L, R
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
    def __init__(self, n, tm_states=5, tm_symbols=2):
        self.n, self.s = n, tm_symbols
        # Goal: a finite state machine that accepts all halting configurations (at least), scanning deterministically then non. Define the state spaces:
        self.S_tape = range(tm_symbols)
        self.Q_tm = range(tm_states)
        self.Q_dfa = range(self.n)
        self.Q_nfa = {NFA_HALT}.union(product(self.Q_dfa, self.Q_tm))
        # The combined output will have...
        self.nfa_args = dict(states=self.Q_nfa.union(self.Q_dfa), input_symbols=set(map(str, self.S_tape)).union(map(ithl, self.Q_tm)), initial_state=DFA_INIT)
        # We use a SAT solver with the following variables representing the problem. (They have accessor methods below.)
        # Also, convention for TM rules: From state f, if bit r read, write bit w, move direction lr, and go to state t.
        self.z3 = Solver()
        self._tm_write = {(f, r): Bool(f'TM_{f}{r}_write') for (f, r) in product(self.Q_tm, self.S_tape)}
        self._tm_right = {(f, r): Bool(f'TM_{f}{r}_right') for (f, r) in product(self.Q_tm, self.S_tape)}
        self._tm_halts = {(f, r): Bool(f'TM_{f}{r}_halts') for (f, r) in product(self.Q_tm, self.S_tape)}
        self._tm_to = {(f, r, t): Bool(f'TM_{f}{r}_to_{t}') for (f, r, t) in product(self.Q_tm, self.S_tape, self.Q_tm)}
        self._dfa = {(q1, r, q2): Bool(f'dfa_{q1}_{r}_{q2}') for (q1, r, q2) in product(self.Q_dfa, self.S_tape, self.Q_dfa)}
        self._nfa = {(x1, r, x2): Bool(f'nfa_{x1}_{r}_{x2}') for (x1, r, x2) in product(self.Q_nfa, self.S_tape, self.Q_nfa)}
        self._accept = {x: Bool(f'accept_{x}') for x in self.Q_nfa}
        # Set up most of the formulae ahead of time.
        self.z3.add(list(chain(self.dfa_rules(), self.nfa_rules(), self.tm_rules(), self.closure_rules())))
        logging.getLogger('\U0001f9ab'*3).info('Ready to solve for DFA size %s', n)

    def dfa(self, q1, r, q2): return self._dfa[q1, r, q2]
    def nfa(self, x1, r, x2): return self._nfa[x1, r, x2]
    def accept(self, x): return self._accept[x]
    def tm_write(self, f, r, w): return self._tm_write[f, r] == bool(w)
    def tm_moves(self, f, r, lr): return self._tm_right[f, r] == (lr==R)
    def tm_halts(self, f, r): return self._tm_halts[f, r]
    def tm_to(self, f, r, t): return self._tm_to[f, r, t]

    def tm_rules(self):
        for f, r in product(self.Q_tm, self.S_tape):
            # Exactly one "to" state per TM transition.
            yield ExactlyOne([self.tm_to(f, r, t) for t in self.Q_tm] + [self.tm_halts(f, r)])

    def machine_code(self, tm):
        for f, r, w, d, t in tm.transitions():
            if t < 0:
                yield self.tm_halts(f, r)
            else:
                yield And(self.tm_write(f, r, w), self.tm_moves(f, r, d), self.tm_to(f, r, t))

    def dfa_rules(self):
        # Determinism: exactly one transition per (state, bit_read)
        for q1, r in product(self.Q_dfa, self.S_tape):
            yield ExactlyOne([self.dfa(q1, r, q2) for q2 in self.Q_dfa])

        # Insensitivity to leading zeros
        yield self.dfa(DFA_INIT, 0, DFA_INIT)

        # Symmetry breaking: if we tabulate [δ(q, r) for q in ... for r in ...], the vertices are seen in order.
        seen = {(q, t): Bool(f'{q} in dfa_table[:{t+1}]') for q in self.Q_dfa for t in range(self.s*self.n+1)}
        for q in self.Q_dfa:
            yield seen[q, 0] == (q==0)
        for q, t in product(range(self.n-1), range(self.s*self.n+1)):
            yield Implies(seen[q+1, t], seen[q, t])
        for q, t in product(self.Q_dfa, range(self.s*self.n)):
            yield seen[q, t+1] == Or(seen[q, t], And(seen[t//self.s, t], self.dfa(t//self.s, t%self.s, q)))

    def nfa_rules(self):
        # Once halted, always halted.
        for s in self.S_tape:
            yield self.nfa(NFA_HALT, s, NFA_HALT)
        # Optimization: HALT shouldn't transition to a normal state. (It doesn't affect acceptance though.)
        for (q, f, b) in product(self.Q_dfa, self.Q_tm, self.S_tape):
                yield Not(self.nfa(NFA_HALT, b, (q, f)))
        # Trailing zeros are not necessary.
        for x, y in product(self.Q_nfa, repeat=2):
            yield Implies(And(self.nfa(x, 0, y), self.accept(y)), self.accept(x))
        # The machine separates the initial configuration from a halted one (and, with the closure rules in place, any potentially halting one).
        yield Not(self.accept((DFA_INIT, TM_INIT)))
        yield self.accept(NFA_HALT)

    def closure_rules(self):
        # If we can reach the configuration after a transition (q1, f) right_tape -> (q2, t) right_tape', we can reach the configuration before.
        for f, r in product(self.Q_tm, self.S_tape):
            halt_rule = self.tm_halts(f, r)
            for q in self.Q_dfa:
                yield Implies(halt_rule, self.nfa((q, f), r, NFA_HALT))
            for w, t in product(self.S_tape, self.Q_tm):
                this = And(self.tm_write(f, r, w), self.tm_to(f, r, t))
                left_rule = And(this, self.tm_moves(f, r, L))
                right_rule = And(this, self.tm_moves(f, r, R))
                for q1, q2 in product(self.Q_dfa, self.Q_dfa):
                    yield Implies(And(right_rule, self.dfa(q1, w, q2)), self.nfa((q1, f), r, (q2, t)))
                    for b1, x1, x2 in product(self.S_tape, self.Q_nfa, self.Q_nfa):
                        yield Implies(And(left_rule, self.dfa(q1, b1, q2), self.nfa((q1, t), b1, x1), self.nfa(x1, w, x2)), self.nfa((q2, f), r, x2))

    def __call__(self, tm, mode, mirrored=False):
        if mirrored:
            tm = reversed(tm)
        solution = (self.z3.check(*self.machine_code(tm)) == sat)
        if mode >= Mode.model and solution:
            solution = self.z3.model()
        if mode >= Mode.nfa and solution:
            transitions  = {q0: {str(r): {q1 for q1 in self.Q_dfa if solution[self.dfa(q0, r, q1)]} for r in self.S_tape} for q0 in self.Q_dfa}
            transitions |= {x0: {str(r): {x1 for x1 in self.Q_nfa if solution[self.nfa(x0, r, x1)]} for r in self.S_tape} for x0 in self.Q_nfa}
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

if __name__ == '__main__':
    from bb_args import ArgumentParser, tm_args
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s: %(message)s')
    ap = ArgumentParser(description='Try to prove the supplied Turing Machines do not halt.', parents=[tm_args()])
    ap.add_argument('-l', help='Size limit for the state machine (NFA) that should gnaw on the tape.', type=int, default=5)
    ap.add_argument('-x', help='Exclude DFAs this small. (Assume we tried already.).', type=int, default=0)
    ap.add_argument('-m', '--mode', help='Level of detail to output.', choices=[m.name for m in Mode], default='re')
    args = ap.parse_args()

    searches = [Search(n, tm_states=args.states, tm_symbols=args.symbols) for n in reversed(range(args.l, args.x, -1))]
    for tm in args.machines:
        assert tm.states == args.states and tm.symbols == args.symbols, 'Oops! TMs with mixed state/symbol counts are not supported.'
        attempts = (search(tm, Mode[args.mode], mirrored=mirrored) for search in searches for mirrored in (False, True))
        solution = next(filter(None, attempts), None)
        if solution:
            print(tm, 'infinite', solution, sep=', ')
        else:
            print(tm, 'undecided', sep=', ')
