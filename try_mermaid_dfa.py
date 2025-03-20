#!/usr/bin/env pypy3
import re
from bb_tm import TM
from dfa_utils import bfs_ordered, line_graph, product
from far_utils import test_solution, optimize_dfa_pair

def parse_mermaid_dfa(file_like, S=2):
    trans = []
    acc = set()
    def reserve(n):
        trans.extend([-1] * (n*S - len(trans)))
    for line in file_like:
        if m := re.match(r'(\d+) -- (\d) --> (\d+)', line):
            q, s, t = map(int, m.groups())
            reserve(q + 1)
            assert trans[q*S + s] == -1, f'Transition specified twice: {(q, s)}'
            trans[q*S + s] = t
        elif m := re.match(r'(\d+)\[_\1_\]', line):
            acc.add(int(m.group(1)))
    trans.extend([-1]*S)  # Dead "-1" state
    return (trans, acc)

def main(tm, mermaid_path, side=1, tries=4, optimize=False):
    with open(mermaid_path) as mermaid_file:
        trans, acc = parse_mermaid_dfa(mermaid_file, tm.symbols)
    dfa = bfs_ordered(trans, tm.symbols)
    for attempt in range(tries):
        if attempt:
            dfa = line_graph(dfa, tm.symbols)
        if test_solution(tm, dfa, side):
            print(f'Worked with {attempt} symbols of added memory.')
            if optimize:
                dfas = [None, None]
                dfas[side] = dfa
                dfas = optimize_dfa_pair(tm, dfas)
                dfa, = filter(None, dfas)
            print('LR'[side], 'DFA:', dfa)
            return
        else:
            print(f'Failed with {attempt} symbols of added memory ({len(dfa)//tm.symbols} states).')


if __name__ == '__main__':
    from bb_args import ArgumentParser, tm_args
    ap = ArgumentParser(description='optimize a DFA proof', parents=[tm_args()])
    ap.add_argument('-r', '--right', help='Use DFA on the right side', action='store_true')
    ap.add_argument('-t', '--tries', help='Try this many times to expand the DFA with a symbol of memory', type=int, default=4)
    ap.add_argument('-o', '--optimize', help='Optimize the returned DFA', action='store_true')
    ap.add_argument('path', help='Path to Mermaid-format DFA file.')
    args = ap.parse_args()
    for tm in args.machines:
        main(tm, args.path, side=args.right, tries=args.tries, optimize=args.optimize)
