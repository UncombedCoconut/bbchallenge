#!/usr/bin/pypy3
# SPDX-FileCopyrightText: 2024 Justin Blanchard <UncombedCoconut@gmail.com>
# SPDX-License-Identifier: Apache-2.0 OR MIT
from argparse import ArgumentParser
from bb_tm import TM
from dataclasses import dataclass
from itertools import chain
from math import isinf
from operator import le, ge
from os import devnull, PathLike
from re import match
from sys import stderr, stdout
from wfa_algebra import Matrix, Series, WeightSet
from wfa_utils import FullCert, ShortCert

@dataclass
class WFACert:
    tm: TM
    initial: Matrix
    transition: dict[str, Matrix]  # keyed by tape and head symbols: '0',...,'A0',...
    final: Matrix

    def verify(self, verbosity=1):
        errors = []
        def test(cond_name, cond, explanation, negate=False):
            if negate:
                cond, cond_name = not cond, f'NOT {cond_name}'
            if not cond:
                errors.append(f'FAIL: {cond_name}\n{explanation}')
                if verbosity: print(self.tm, errors[-1], file=stderr)
            elif verbosity > 1: print(self.tm, f'PASS: {cond_name}\n{explanation}', file=stderr)
        def test_fwd(lhs_name, lhs, rhs_name, rhs, negate=False):
            test(f'{lhs_name} <= {rhs_name}', lhs <= rhs, f' - {lhs_name} = {lhs}\n - {rhs_name} = {rhs}', negate=negate)
        S = range(self.tm.symbols)

        test_fwd('<^|', self.initial, '<^||0|', self.initial * self.transition['0'])
        test_fwd('|$>', self.final, '|0||$>', self.transition['0'] * self.final)
        for f, r, w, d, t in self.tm.transitions():
            fr = chr(65+f) + str(r)
            if t < 0:
                test_fwd(f'|{fr}|', self.transition[fr], '|Z|', self.transition['Z'])
                continue
            for s in S:
                ts = chr(65+t) + str(s)
                if d: # left
                    test_fwd(f'|{s}||{fr}|', self.transition[f'{s}'] * self.transition[fr], f'|{ts}|{w}|', self.transition[ts] * self.transition[f'{w}'])
                else: # right
                    test_fwd(f'|{fr}||{s}|', self.transition[fr] * self.transition[f'{s}'], f'|{w}|{ts}|', self.transition[f'{w}'] * self.transition[ts])
        for s in S: test_fwd(f'|Z||{s}|', self.transition['Z'] * self.transition[f'{s}'], '|Z|', self.transition['Z'])
        for s in S: test_fwd(f'|{s}||Z|', self.transition[f'{s}'] * self.transition['Z'], '|Z|', self.transition['Z'])
        test_fwd('<^|A0|$>', self.initial * self.transition['A0'] * self.final, '<^|Z|$>', self.initial * self.transition['Z'] * self.final, negate=True)
        if verbosity and not errors:
            print(self.tm, 'PASS', file=stderr)
        return not errors

    def desc_algebraic(self):
        return '\n'.join(chain(self._lines_common(), self._lines_algebraic()))

    def desc_graphical(self):
        return '\n'.join(chain(self._lines_common(), self._lines_graphical()))

    def _lines_common(self):
        yield f'TM {self.tm}'
        if self.tm.seed: yield f'Seed {self.tm.seed}'
        yield f'States {" ".join(self.transition["0"].l_basis)}'

    def _lines_algebraic(self):
        yield f'<^| = {self.initial}'
        for infix, trans in self.transition.items():
            yield f'|{infix}| = {trans}'
        yield f'|$> = {self.final}'

    def _lines_graphical(self):
        monos = lambda nonzero_coef: getattr(nonzero_coef, 'monos', {0})

        for i, (name, c0i) in enumerate(zip(self.initial.r_basis, self.initial.c[0])):
            if c0i: yield f'Initial {name} {monos(c0i)}'
        for infix, trans in self.transition.items():
            for l_name, cl in zip(trans.l_basis, trans.c):
                for r_name, clr in zip(trans.r_basis, cl):
                    if clr: yield f'Edge {l_name} {infix} {r_name} {monos(clr)}'
        for i, (name, ci) in enumerate(zip(self.final.l_basis, self.final.c)):
            if ci[0]: yield f'Final {name} {monos(ci[0])}'

    @classmethod
    def parse(cls, file, verbosity=1):
        if isinstance(file, (str, bytes, PathLike)): file = open(file)
        new = cls(None, None, {}, None)
        for i, line in enumerate(file):
            # Allow comments
            line = line.partition('#')[0].strip()
            if not line: continue
            elif m := match(r'TM\s+([A-Z0-9-_]+)', line):
                if new.tm is not None: yield new
                new = cls(TM.from_text(m.group(1)), None, {}, None)
            elif m := match(r'Seed\s+(\d+)', line):
                  new.tm.seed = int(m.group(1))
            elif m := match(r'States\s+(.*)', line):
                states = m.group(1).split()
                new.initial = Matrix(('',), states)
                new.transition = {infix: Matrix(states, states) for infix in [str(s) for s in range(new.tm.symbols)] + [chr(65+q) + str(s) for q in range(new.tm.states) for s in range(new.tm.symbols)] + ['Z']}
                new.final = Matrix(states, ('',))
            elif m := match(r'<\^\| = (.*)', line):
                new.initial = Matrix.from_text(m.group(1), new.initial.l_basis, new.initial.r_basis)
            elif m := match(r'\|([^|]*)\| = (.*)', line):
                new.transition[m.group(1)] = Matrix.from_text(m.group(2), new.transition['0'].l_basis)
            elif m := match(r'\|\$> = (.*)', line):
                new.final = Matrix.from_text(m.group(1), new.final.l_basis, new.final.r_basis)
            elif m := match(r'Initial\s+([^\s]+)\s+(.*)', line):
                new.initial[0, new.initial.r_basis.index(m.group(1))] += Series(WeightSet.from_text(m.group(2)))
            elif m := match(r'Edge\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+(.*)', line):
                trans = new.transition[m.group(2)]
                trans[trans.l_basis.index(m.group(1)), trans.r_basis.index(m.group(3))] += Series(WeightSet.from_text(m.group(4)))
            elif m := match(r'Final\s+([^\s]+)\s+(.*)', line):
                new.final[new.final.l_basis.index(m.group(1)), 0] += Series(WeightSet.from_text(m.group(2)))
            elif verbosity:
                print('Input line not understood:', line, file=stderr)
        if new.tm is not None: yield new

    @classmethod
    def from_iijil_cert(cls, cert):
        head_symbols = [chr(65+q) + str(s) for q in range(cert.tm.states) for s in range(cert.tm.symbols)]
        l0, lt = matrix_form(cert.wfas[0], symbols=cert.tm.symbols, state_prefix='L')
        r0, rt = matrix_form(cert.wfas[1], symbols=cert.tm.symbols, state_prefix='R')
        initial = l0.oplus(r0.zero())
        final = l0.zero().oplus(r0).transpose(conjugate=False)

        transition = {s: lt[s].oplus(rt[s].transpose(conjugate=False)) for s in map(str, range(cert.tm.symbols))}
        sum_sym_trans = sum(transition.values()).geometric_series()
        transition |= {infix: transition['0'].zero() for infix in head_symbols + ['Z']}
        # Turn the closure conditions into a giant linear inequality V >= M*V in the coefficients of the transitions |A0|,....
        # A TM transition f,r -> w,L,t gives us the inequality |s||fr| <= |ts||w| and thus |ts| >= |s||fr||w*| when the right-side WFA is deterministic.
        coef_basis = [f'{l}.{head}.{r}' for l in l0.r_basis for head in head_symbols for r in r0.r_basis]
        M = Matrix(coef_basis, coef_basis)
        for f, r, w, d, t in cert.tm.transitions():
            if t < 0: continue
            fr = chr(65+f) + str(r)
            for s in range(cert.tm.symbols):
                ts = chr(65+t) + str(s)
                # d indicates if it's a left rule. Either way, we get a bound of the form |ts| >= opL |fr| opR.
                opL, opR = (transition[f'{s}'], transition[f'{w}'].conjugate()) if d else (transition[f'{w}'].conjugate(), transition[f'{s}'])
                # And now: <lout|ts|rout> >= <lout|opL| |lin><lin| |fr| |rin><rin| |opR|rout>
                for lout, opLlout in zip(opL.l_basis, opL.c):
                    for rin, opRrin in zip(opR.l_basis, opR.c):
                        for lin, coefL in zip(opL.r_basis, opLlout):
                            for rout, coefR in zip(opR.r_basis, opRrin):
                                try: # If we aren't looking at nonsense coefficients that will remain zero, like <R?|fr|L?>...
                                    M[coef_basis.index(f'{lout}.{ts}.{rout}'), coef_basis.index(f'{lin}.{fr}.{rin}')] += coefL * coefR
                                except ValueError: pass
        # V >= M*V implies V = (1 + M + M^2 + ...) V; take the minimum such fixed point such that <L0|A0|R0> >= 1.
        necessary = coef_basis.index('L0.A0.R0')
        for lhr, fixpoint_for in zip(coef_basis, M.geometric_series().c):
            l, h, r = lhr.split('.')
            th = transition[h]
            th[th.l_basis.index(l), th.r_basis.index(r)] += fixpoint_for[necessary]

        for f, r, w, d, t in cert.tm.transitions():
            if t < 0:
                transition['Z'] += transition[chr(65+f) + str(r)]
        transition['Z'] = sum_sym_trans * transition['Z'] * sum_sym_trans

        return cls(cert.tm, initial, transition, final)

def matrix_form(wfa, symbols, state_prefix):
    states = [f'{state_prefix}{q}' for q in range(len(wfa.t) // symbols)]
    start = Matrix(('',), states, [[1] + [0 for _ in states[1:]]])
    transition = {f'{s}': Matrix(states, states) for s in range(symbols)}
    for qs, (t, w) in enumerate(zip(wfa.t, wfa.w)):
        q, s = divmod(qs, symbols)
        transition[f'{s}'][q, t] += Series((w,))
    return start, transition

if __name__ == '__main__':
    ap = ArgumentParser(description='Verify WFA proofs.')
    ap.add_argument('-v', '--verbosity', metavar='N', help='verbosity level: at 1, report reasons for failure; at 2, report all checks.', type=int, default=1)
    ap.add_argument('-g', '--good', help='output location for certs judged to be good', default='-')
    ap.add_argument('-b', '--bad', help='output location for certs judged to be bad', default=devnull)
    ap.add_argument('-t', '--type', help='choice of [a]lgebraic or [g]raphical description, or just [s]eed or [t]ext of TM', default='algebraic')
    ap.add_argument(      '--fc', help='import these from MITMWFAR "FullCert" format', nargs='*', default=())
    ap.add_argument(      '--sc', help='import these from MITMWFAR "ShortCert" format', nargs='*', default=())
    ap.add_argument('files', nargs='*')
    args = ap.parse_args()

    native = chain.from_iterable(map(WFACert.parse, args.files))
    foreign = chain(map(FullCert.parse, args.fc), map(ShortCert.parse, args.sc))
    converted = (WFACert.from_iijil_cert(fc) for fc_file in foreign for fc in fc_file)

    good = stdout if args.good == '-' else open(args.good, 'a')
    bad = stdout if args.bad == '-' else open(args.bad, 'a')

    match args.type[:1].lower():
        case 'a':
            def output(cert, file):
                print(cert.desc_algebraic(), file=file)
        case 'g':
            def output(cert, file):
                print(cert.desc_graphical(), file=file)
        case 's':
            def output(cert, file):
                print(cert.tm.seed, file=file)
        case 't':
            def output(cert, file):
                print(cert.tm, file=file)
        case _:
            raise RuntimeError(f'Unrecognized output type: {args.type!r}')

    try:
        for cert in chain(native, converted):
            passed = cert.verify(args.verbosity)
            output(cert, good if passed else bad)
    finally:
        if good is not stdout: good.close()
        if bad is not stdout: bad.close()
