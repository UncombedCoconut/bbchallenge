#!/usr/bin/pypy3
# SPDX-FileCopyrightText: 2024 Justin Blanchard <UncombedCoconut@gmail.com>
# SPDX-License-Identifier: Apache-2.0 OR MIT
from dfa_utils import reachable_states, redirect
from wfa_utils import ShortCert, WFA
from argparse import ArgumentParser
from collections import deque
from itertools import permutations
from subprocess import Popen, PIPE
import sys

def main():
    from bb_db import DB
    db = DB()
    tricky_beavers = [320969, 321080, 472097, 523280, 572659, 1465912, 8210683, 8210684, 8210685, 8226493, 10818491, 10818510, 7763480, 8120967, 10756090, 11017998, 11018487, 4817065, 5377178, 7138511, 12699987, 12781955]
    with open('REWRITE_ShortCerts.txt', 'w') as rewrite:
        for sc in ShortCert.parse('SolvedShortCerts.txt'):
            for seed in tricky_beavers:
                tm = db[seed]
                if str(sc.tm) == str(tm): sc.tm = tm
            print(sc.tm.seed, len(sc.wfas[0].t)//tm.symbols, len(sc.wfas[1].t)//tm.symbols)
            print(sc, file=rewrite)


def optimize_cert(sc, popen, verbose=True):
    dfas = [sc.wfas[side].t for side in range(2)]
    S = sc.tm.symbols
    sc_reduced = ShortCert(sc.tm, sc.wfas)
    sort_key = lambda dq1q2: (_size_aggr(reachable_states(redirect(dfas[dq1q2[0]], dq1q2[1], dq1q2[2]), S), len(sc_reduced.wfas[1-dq1q2[0]].t)//S), dq1q2[2], dq1q2[1])
    identifications = [(side, q1, q2) for side in range(2) for (q1, q2) in permutations(range(len(sc.wfas[side].t)//S), 2)]
    identifications.sort(key=sort_key)

    if verbose:
        print(_cert_size_text(sc), end='', file=sys.stderr)
    for iident, (side, q1, q2) in enumerate(identifications):
        q_unreduced = redirect(dfas[side], q1, q2)
        cand_sc_reduced = ShortCert(sc.tm, list(sc_reduced.wfas))
        cand_sc_reduced.wfas[side] = ordered_wfa(WFA(q_unreduced, sc.wfas[side].w), S)
        if len(cand_sc_reduced.wfas[side].t) >= len(sc_reduced.wfas[side].t): continue
        status = f'?{_cert_size_text(cand_sc_reduced)}|{(1+iident)*100/len(identifications):.3f}%'
        if verbose:
            print(status, end='', file=sys.stderr, flush=True)
            print('\b \b'*len(status), end='', flush=False, file=sys.stderr)
        if _test_cert(cand_sc_reduced, popen):
            dfas[side] = q_unreduced
            sc_reduced = ShortCert(sc.tm, tuple(cand_sc_reduced.wfas))
            identifications[iident+1:] = sorted(identifications[iident+1:], key=sort_key)
            if verbose:
                print(f'>{_cert_size_text(sc_reduced)}', end='', file=sys.stderr, flush=True)
    if verbose:
        print(f' => {_cert_size_text(sc_reduced)}', file=sys.stderr, flush=True)
    return sc_reduced

def ordered_wfa(wfa, S=2):
    '''Return an equivalent WFA with states ordered by breadth-first search (and unreachable states stripped).'''
    n, used = len(wfa.t)//S, 1
    state_id = [None]*n
    state_id[0] = 0
    trans = []
    weigh = []
    bfs_q = deque((0,))
    while bfs_q:
        for _ in range(S):
            trans.append(None)
            weigh.append(None)
        q0 = bfs_q.popleft()
        i0 = state_id[q0]
        for s in range(S):
            q1 = wfa.t[S*q0+s]
            i1 = state_id[q1]
            if i1 is None:
                i1 = state_id[q1] = used
                used += 1
                bfs_q.append(q1)
            trans[S*i0+s] = i1
            weigh[S*i0+s] = wfa.w[S*q0+s]
    return WFA(trans, weigh)

def _test_cert(sc, popen):
    # Pass the program the cert we want to check, followed by a blindingly obvious one, so that we get at least one line back.
    print(sc, file=popen.stdin)
    print('0RA_0RA\n0,0\n0,0', file=popen.stdin, flush=True)
    response = popen.stdout.readline().strip()
    if response == '0RA_0RA': return False
    assert response == str(sc.tm), f'Unexpected MITMWFAR reply {response!r}'
    assert popen.stdout.readline().strip() == '0RA_0RA', f'Unexpected MITMWFAR reply {response!r}'
    return True

def _cert_size(sc):
    return _size_aggr(len(sc.wfas[0].t)//sc.tm.symbols, len(sc.wfas[1].t)//sc.tm.symbols)

def _size_aggr(lsize, rsize):
    return lsize * rsize

def _cert_size_text(sc):
    return '{}x{}'.format(len(sc.wfas[0].t)//sc.tm.symbols, len(sc.wfas[1].t)//sc.tm.symbols)

if __name__ == '__main__':
    ap = ArgumentParser(description='optimize a ShortCerts file')
    ap.add_argument('-p', '--program', help='path to the MITMWFAR executable', default='./MITMWFAR')
    ap.add_argument('files', help='one or more ShortCerts files', nargs='*', default=['SolvedShortCerts.txt'])
    args = ap.parse_args()
    with Popen([args.program, '-sc', '-cores', '1'], stdin=PIPE, stdout=PIPE, text=True) as popen:
        for file in args.files:
            for sc in ShortCert.parse(file):
                score = _cert_size(sc)
                while True:
                    sc = optimize_cert(sc, popen=popen)
                    if (new_score := _cert_size(sc)) == score:
                        break
                    score = new_score
                print(sc)
