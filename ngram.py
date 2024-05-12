#!/usr/bin/pypy3
# SPDX-FileCopyrightText: 2022 Justin Blanchard <UncombedCoconut@gmail.com>
# SPDX-License-Identifier: Apache-2.0 OR MIT
from bb_tm import HALT, TM
from collections import defaultdict, deque
import json, sys, tqdm

def decide(aug_tm, radius, max_contexts, return_proof=False):
    # Store n-grams in MitM-order (outside-in), local contexts as (state, sym, (L n-gram, R n-gram)). Use a scratch tape centered at C (head loc before step).
    tape = [aug_tm.default_symbol()] * (2*radius+3)
    C = radius + 1
    stack = [(aug_tm.default_state(), tape[C], (tuple(tape[C-radius:C]), tuple(tape[C+radius:C:-1])))]
    contexts = set()
    n_gram_ext = [defaultdict(set, {n_gram[1:]: {n_gram}}) for n_gram in stack[0][2]]
    context_ext = [defaultdict(set) for side in range(2)]
    while stack:
        context = stack.pop()
        if not add_is_new(contexts, context): continue
        if len(contexts) > max_contexts: return False
        q0, s0, lr0 = context
        lri = lr1 = [None, None]  # Successor n-gram pair (named twice for the uses below).
        tape[C-radius:C], tape[C+radius:C:-1] = lr0
        tape[C], d, q1 = aug_tm.transition(q0, s0)
        if q1 == HALT: return False
        dx = 1 - 2*d
        s1 = tape[C+dx]
        pop_key = tuple(tape[C+radius*dx:C+dx:-dx])
        # Push
        n_gram = lr1[d] = tuple(tape[C-(radius-1)*dx:C+dx:dx])
        push_key = n_gram[1:]
        if add_is_new(n_gram_ext[d][push_key], n_gram): # Need to finish incomplete searches from previously extended local contexts that match.
            for qi, si, lri[1-d] in list(context_ext[d][push_key]):
                stack.append((qi, si, tuple(lri)))
        # Pop
        if add_is_new(context_ext[1-d][pop_key], (q1, s1, lr1[d])): # Dually, new partial local context may match previously seen n-grams.
            for lr1[1-d] in list(n_gram_ext[1-d][pop_key]):
                stack.append((q1, s1, tuple(lr1)))
    if return_proof: return [to_dfa(aug_tm, set.union(*n_grams.values())) for n_grams in n_gram_ext]
    return True

def to_dfa(tm, n_grams):
    n_grams = list(n_grams)
    transition_map = [defaultdict(list) for _ in n_grams]  # [n_gram_id]["base" TM symbol] -> list of possible n_gram IDs
    for n_gram, n_gram_trans in zip(n_grams, transition_map):
        # FIXME: no need for a quadratic loop here.
        for i_dest, dest in enumerate(n_grams):
            if n_gram[1:] == dest[:-1]:
                n_gram_trans[tm.base_symbol(dest[-1])].append(i_dest)
    all_zero_n_grams = frozenset(i for i, n_gram in enumerate(n_grams) if all(tm.base_symbol(s) == 0 for s in n_gram))

    state_id = {all_zero_n_grams: 0}
    trans = []
    bfs_q = deque(state_id)
    while bfs_q:
        for _ in range(tm.symbols):
            trans.append(None)
        q0 = bfs_q.popleft()
        i0 = state_id[q0]
        for s in range(tm.symbols):
            q1 = frozenset([dest_id for gram_id in q0 for dest_id in transition_map[gram_id][s]])
            try:
                i1 = state_id[q1]
            except KeyError:
                i1 = state_id[q1] = len(state_id)
                bfs_q.append(q1)
            trans[tm.symbols * i0 + s] = i1
    return trans

def to_cert(tm, dfas):
    cert = dict(tm=str(tm), seed=tm.seed, cert_type='FAR', dfas=dfas)
    if tm.seed is None: del cert['seed']
    return json.dumps(cert)

def add_is_new(s, e):
    before = len(s)
    s.add(e)
    return before != len(s)

class AugTM:
    def default_state(self):
        return self.base.default_state()
    def default_symbol(self):
        return self.base.default_symbol()
    def base_symbol(self, s):
        return self.base.base_symbol(s)
    @property
    def symbols(self):
        return self.base.symbols

class AugWriteLog(AugTM):
    __slots__ = ('base', 'log_length')

    def __init__(self, base, log_length):
        self.base = base
        self.log_length = log_length

    def transition(self, f, r):
        base_r = r[0] if r else self.base.default_symbol()
        w, d, t = self.base.transition(f, base_r)
        return (w,) + r[:self.log_length-1], d, t

    def default_symbol(self):
        return ()

    def base_symbol(self, s):
        return self.base.base_symbol(s[0]) if s else 0

class AugFR(AugTM):
    __slots__ = ('base')

    def __init__(self, base):
        self.base = base

    def transition(self, f, r):
        base_r = r[0] if r else self.base.default_symbol()
        w, d, t = self.base.transition(f, base_r)
        return (w, f, base_r), d, t

    def default_symbol(self):
        return None

    def base_symbol(self, s):
        return self.base.base_symbol(s[0]) if s else 0

class AugLRU(AugTM):
    __slots__ = ('base')

    def __init__(self, base):
        self.base = base

    def transition(self, f, r):
        w, d, t = self.base.transition(f[0], r)
        if t != HALT:
            try:
                i = f[1].index((f[0], r))
                t = (t, ((f[0], r),) + f[1][:i] + f[1][i+1:])
                #t = (t, ((f[0], r),) + f[1][:i])
            except ValueError:
                t = (t, ((f[0], r),) + f[1])
        return w, d, t

    def default_state(self):
        return (self.base.default_state(), ())

class AugPosModN(AugTM):
    __slots__ = ('base', 'n')

    def __init__(self, base, n):
        self.base = base
        self.n = n

    def transition(self, f, r):
        w, d, t = self.base.transition(f[0], r)
        dx = 1 - 2*d # R->+1, L->-1
        if t != HALT:
            t = (t, (f[1] + dx) % self.n)
        return w, d, t

    def default_state(self):
        return (self.base.default_state(), 0)


if __name__ == '__main__':
    from bb_args import ArgumentParser, tm_args
    ap = ArgumentParser(parents=[tm_args()])
    ap.add_argument('-p', '--proof', help='Emit certificates, not just decisions.', action='store_true')
    ap.add_argument('-r', '--radius', help='radius', type=int, default=4)
    ap.add_argument('-c', '--contexts', help='max local contexts', type=int, default=10**6)
    ap.add_argument('-H', '--fr-history', help='Augment symbols with the head config (from_state, read_sym).', action='store_true')
    ap.add_argument('-l', '--fr-lru', help='Augment state with an LRU-list of recent head configs (from_state, read_sym).', action='store_true')
    ap.add_argument('-x', metavar='N', help='Augment state with x coordinate mod N.', type=int, default=0)
    ap.add_argument('-w', metavar='N', help='Augment symbols with a write-log of length n.', type=int, default=0)
    args = ap.parse_args()

    todo = args.machines
    for radius in range(2, args.radius+1):
        todo_old, todo = todo, []
        for tm in tqdm.tqdm(todo_old, desc=f'{radius=}'):
            aug = tm
            if args.x:
                aug = AugPosModN(aug, args.x)
            if args.fr_lru:
                aug = AugLRU(aug)
            if args.fr_history or args.fr_lru:
                aug = AugFR(aug)
            if args.w:
                aug = AugWriteLog(aug, args.w)
            if (result := decide(aug, radius, args.contexts, return_proof=args.proof)):
                print(to_cert(tm, result) if args.proof else tm)
            else:
                todo.append(tm)
    print('->', len(todo), 'to go', file=sys.stderr)
