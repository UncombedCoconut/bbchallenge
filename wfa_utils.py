#!/usr/bin/python3
# SPDX-FileCopyrightText: 2024 Justin Blanchard <UncombedCoconut@gmail.com>
# SPDX-License-Identifier: Apache-2.0 OR MIT
from argparse import ArgumentParser
from bb_tm import TM
from dataclasses import dataclass
from itertools import islice #batched
from os import PathLike
from PIL import ImageShow
import subprocess

@dataclass
class WFA:
    t: list[int]
    w: list[int]

    @classmethod
    def from_text(cls, text):
        wfa = cls([], [])
        for by_q in text.strip().split('_'):
            for by_qs in by_q.split(';'):
                t, qw = map(int, by_qs.split(','))
                wfa.t.append(t)
                wfa.w.append(qw)
        return wfa

    def to_text(self, tm_symbols):
        return '_'.join([';'.join([f'{t},{w}' for (t, w) in group]) for group in batched(zip(self.t, self.w), tm_symbols)])

@dataclass
class ShortCert:
    tm: TM
    wfas: tuple[WFA, WFA]

    @classmethod
    def parse(cls, file):
        if isinstance(file, (str, bytes, PathLike)): file = open(file)
        for i, line in enumerate(file):
            match i%3:
                case 0: tm = TM.from_text(line.strip())
                case 1: l = WFA.from_text(line)
                case 2: yield cls(tm, (l, WFA.from_text(line)))

    def __str__(self):
        return f'{self.tm}\n{self.wfas[0].to_text(self.tm.symbols)}\n{self.wfas[1].to_text(self.tm.symbols)}'

def sink(dfa, symbols):
    sinks = [q for q in range(1, len(dfa)//symbols) if all(t==q for t in dfa[q*symbols:(q+1)*symbols])]
    if len(sinks) == 1: return sinks[0]

def add_weight(wfa, q, dw, tm):
    for qs, t in enumerate(wfa.t):
        if wfa.t[qs] == q:
            wfa.w[qs] += dw
    for s in range(tm.symbols):
        wfa.w[q*tm.symbols+s] -= dw

def zero_bfs_tree(wfa, tm):
    visited = set()
    for qs, t in enumerate(wfa.t):
        if t not in visited:
            add_weight(wfa, t, -wfa.w[qs], tm)
            visited.add(t)

def save_dot(short_cert, filename):
    S = short_cert.tm.symbols
    size = [len(wfa.t)//S for wfa in short_cert.wfas]
    rip = [sink(wfa.t, S) for wfa in short_cert.wfas]
    with open(filename, 'w') as f:
        print('digraph F {', file=f)
        print('    newrank=true;', file=f)
        print('    rankdir=LR;', file=f)
        for side in range(2):
            if side: print(file=f)
            lr = 'LR'[side]
            #print(f'    subgraph cluster{lr} {{', file=f)
            for q in range(size[side]):
                if q != rip[side]:
                    print(f'        "{lr}{q}" [label="{q}"]', file=f)
            for qs, (t, w) in enumerate(zip(short_cert.wfas[side].t, short_cert.wfas[side].w)):
                q, s = divmod(qs, S)
                if q == rip[side] or t == rip[side]: continue
                match side:
                    case 0: print(f'        "L{q}" -> "L{t}" [label="{s}", color="{COLOR[w]}"]', file=f)
                    case 1: print(f'        "R{t}" -> "R{q}" [dir=back, label="{s}", color="{COLOR[w]}"]', file=f)
            #print('    }', file=f)
        max_non_sink = [size_i - 1 - int(size_i-1==rip_i) for size_i, rip_i in zip(size, rip)]
        print(f'    {{ rank=same; "L{max_non_sink[0]}"; "R{max_non_sink[1]}" }}', file=f)
        print('}', file=f)

# I'll regret this eventually.
COLOR = {
        8: 'violet', 7: 'violet', 6: 'violet', 5: 'violet',
        -8: 'blue', -7: 'blue', -6: 'blue', -5: 'blue',
        -4: 'red4', -3: 'red3', -2: 'red2', -1: 'red1', 0: 'black', 1: 'green1', 2: 'green2', 3: 'green3', 4: 'green4'}

# This is a backport of a Python 3.12 standard library function, itertools.batched.
def batched(it, n):
    while batch := tuple(islice(it, n)): yield batch

if __name__ == '__main__':
    ap = ArgumentParser(description='Visualize a MITMWFAR short certificate.')
    ap.add_argument('-z', '--zero', help='Zero the weights along each the BFS tree of each DFA.', action='store_true')
    ap.add_argument('files', nargs='*')
    args = ap.parse_args()
    for file in args.files:
        for short_cert in ShortCert.parse(file):
            base = f'wfa_pair_{short_cert.tm}'
            if args.zero:
                for wfa in short_cert.wfas:
                    zero_bfs_tree(wfa, short_cert.tm)
                # HACK
                pairs = tuple([(tuple(wfa.t), tuple(wfa.w)) for wfa in short_cert.wfas])
                print(f'{pairs=}')
            save_dot(short_cert, f'{base}.dot')
            subprocess.check_call(['dot', '-Tpng', '-O', f'{base}.dot'])
            ImageShow._viewers[0].show_file(f'{base}.dot.png')
