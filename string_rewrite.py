#!/usr/bin/pypy3
# SPDX-FileCopyrightText: 2022 Justin Blanchard <UncombedCoconut@gmail.com>
# SPDX-License-Identifier: Apache-2.0 OR MIT
from bbchallenge import get_machine_i, ithl, g
from collections import Counter
from dataclasses import dataclass
import re


def flip(bit_char):
    return chr(ord(bit_char)^1)


@dataclass
class Word:
    '''
    We model a TM's state TM as a word in the language [01]*[a-eA-E][01]*.
    Lower-case a-e represent the TM head in state A-E on top of a 0 bit.
    Upper-case A-E represent the TM head in state A-E on top of a 1 bit.
    A "Word" is just a string (with designated left/state/right components).
    Finite tape segments which include the head are Words.
    TM states are Words modulo leading/trailing zeros.
    '''
    l: str = ''
    s: str = ''
    r: str = ''

    def __str__(self):
        return f'{self.l}{self.s}{self.r}'

    def __contains__(self, word):
        return self.l.endswith(word.l) and self.s == word.s and self.r.startswith(word.r)

    def slice(self, l_len, r_len):
        return Word(self.l[len(self.l)-l_len:], self.s, self.r[:r_len])

    def l_bitdrop(self):
        return Word(self.l[1:], self.s, self.r)

    def r_bitdrop(self):
        return Word(self.l, self.s, self.r[:-1])

    def l_bitflip(self):
        return Word(flip(self.l[0]) + self.l[1:], self.s, self.r)

    def r_bitflip(self):
        return Word(self.l, self.s, self.r[:-1] + flip(self.r[-1]))

    def __add__(self, bits):
        return Word(self.l, self.s, self.r + bits)

    def __radd__(self, bits):
        return Word(bits + self.l, self.s, self.r)

    def __reversed__(self):
        return Word(self.r[::-1], self.s, self.l[::-1])

    def from_str(string):
        return Word(*re.match('^([01]*)([a-eA-E])([01]*)$', string).groups())


@dataclass
class Rewrite:
    '''
    Simply a substitution "f"rom self.f "t"o self.t.
    The devil is in the details of application (is the input Word a tape-state, implicitly zero extended?)
    and of composition (may we restrict to a common domain?).
    '''
    f: Word
    t: Word

    def apply_to(self, word, as_tape=False):
        ''' Return the word formed by applying the substitution once (if it applies). If as_tape, the word implicitly has zeros on both sides.'''
        if self.f in word:
            return Word(word.l[:len(word.l)-len(self.f.l)] + self.t.l, self.t.s, self.t.r + word.r[len(self.f.r):])
        elif as_tape:
            n_lpad = max(0, len(self.f.l) - len(word.l))
            n_rpad = max(0, len(self.f.r) - len(word.r))
            padded = Word('0'*n_lpad + word.l, word.s, word.r + '0'*n_rpad)
            out = self.apply_to(padded)
            if out is padded:
                return word
            else:
                return Word(out.l.lstrip('0'), out.s, out.r.rstrip('0'))
        return word

    def then(self, rhs, total_only=False):
        ''' Return the composition of two rewrites. If total_only, the 2nd rewrite is requried to apply to all outputs of the 1st.
            Otherwise, return the rewrite which applies both substiitutions in order *where possible*.
            If the rewrites don't compose, return None.'''
        if self.t.s != rhs.f.s:
            return
        if self.t.l.endswith(rhs.f.l):
            lpad = ''
        elif not total_only and rhs.f.l.endswith(self.t.l):
            lpad = rhs.f.l[:len(rhs.f.l)-len(self.t.l)]
        else:
            return

        if self.t.r.startswith(rhs.f.r):
            rpad = ''
        elif not total_only and rhs.f.r.startswith(self.t.r):
            rpad = rhs.f.r[len(self.t.r):]
        else:
            return

        padded_f = Word(l=lpad+self.f.l, s=self.f.s, r=self.f.r+rpad)
        padded_t = Word(l=lpad+self.t.l, s=self.t.s, r=self.t.r+rpad)
        return Rewrite(padded_f, rhs(padded_t))

    __call__ = apply_to
    __mul__ = then  # Support f*g to meaning f followed by g (g(f(_)))... hopefully a good idea?

    def mirror(self):
        return Rewrite(reversed(self.f), reversed(self.t))

    def __str__(self):
        return f'{self.f}\u2192{self.t}'


class RewriteSystem:
    '''
    A "deterministic rewrite system" over the language of tape words [01]*[a-eA-E][01]*.
    The possible tape states are partitioned into those matching exactly one of a finite set of Words, which are:
    1. halting, 2. unreachable, 3. cycling, or 4. the "from" side of a Rewrite.
    '''

    def __init__(self, machine):
        self.special_words = {'halting': [], 'cycling': [], 'unreachable': []}
        self.rewrites = []
        self.start = Word(s='a')
        for rewrite in self.read(machine):
            self._add_rule(rewrite)

    def __str__(self):
        word_lists = [f'{typ} (' + '|'.join(map(str, words)) + ')' for (typ, words) in self.special_words.items()]
        return ', '.join([f'TM Rewrite System: start {self.start}', *word_lists,
                'rules: ' + '\n    '.join([''] + [str(rewrite) for rewrite in self.rewrites])])

    def mirror(self):
        srs = RewriteSystem.__new__(RewriteSystem)
        srs.special_words = {typ: [reversed(w) for w in words] for (typ, words) in self.special_words.items()}
        srs.rewrites = [rw.mirror() for rw in self.rewrites]
        srs.start = reversed(self.start)
        return srs

    def read(self, machine):
        for i in range(5):
            for j, case in enumerate((str.lower, str.upper)):
                s = ithl(i)
                write = machine[6*i+3*j]
                move = g(machine[6*i+3*j+1])
                goto = ithl(machine[6*i+3*j+2]-1)
                if goto >= 'A':
                    if move == 'L':
                        yield Rewrite(Word(l='0', s=case(s)), Word(s=goto.lower(), r='01'[write]))
                        yield Rewrite(Word(l='1', s=case(s)), Word(s=goto.upper(), r='01'[write]))
                    else:
                        yield Rewrite(Word(r='0', s=case(s)), Word(s=goto.lower(), l='01'[write]))
                        yield Rewrite(Word(r='1', s=case(s)), Word(s=goto.upper(), l='01'[write]))
                else:
                    halting_word = Word(s=case(s))
                    self._add_word('halting', halting_word)
                    for k in reversed(range(len(self.rewrites))):
                        if halting_word in self.rewrites[k].t:
                            self._add_word('halting', self.rewrites[k].f)
                            del self.rewrites[k]

    def _add_word(self, typ, word):
        ''' Add the given word to the given special_words list. To avoid redundancy, merge with adjacent words, e.g. 0e0|0e1 -> 0e. '''
        lst = self.special_words[typ]
        # To avoid redundancy, merge with existing words.
        while True:
            try:
                lst.remove(word.l_bitflip())
                word = word.l_bitdrop()
                continue
            except (IndexError, ValueError):
                pass
            try:
                lst.remove(word.r_bitflip())
                word = word.r_bitdrop()
                continue
            except (IndexError, ValueError):
                break
        lst.append(word)

    def _add_rule(self, rewrite):
        ''' Add the given rewrite to the rule list. Ensure that each rule has had all total post-compositions applied already. '''
        self.rewrites.append(rewrite)
        self._try_post_compositions(-1)

        # See if any older entries now have post-compositions to consider.
        i = 0
        while i < len(self.rewrites):
            try_before = self.rewrites[i]
            if (comp := try_before.then(rewrite, total_only=True)):
                self.rewrites[i] = comp
                if self._try_post_compositions(i):
                    i -= 1
            i += 1

    def _try_post_compositions(self, i):
        ''' Follow up self.rewrites[i] with other rules (total post-compositions only) for as long as possible.
            This may lead to a halting or cycling behavior. If so, delete the rewrite and mark its domain as appropriate.
            Return True if we had to do that, False otherwise. '''
        # ASSUMPTION: all rewrites are word-length-preserving (true for TM rules), so an infinite loop will actually revisit a word.
        rewrite = self.rewrites[i]
        output_words = [rewrite.t]
        while True:
            for typ in 'halting', 'cycling':
                if any(word in rewrite.t for word in self.special_words[typ]):
                    self._add_word(typ, rewrite.f)
                    del self.rewrites[i]
                    return True
            for try_after in self.rewrites:
                if (comp := rewrite.then(try_after, total_only=True)):
                    rewrite = comp
                    if rewrite.t in output_words:
                        self._add_word('cycling', rewrite.f)
                        del self.rewrites[i]
                        return True
                    output_words.append(rewrite.t)
                    break
            else:
                break
        self.rewrites[i] = rewrite
        return False

    def prune(self):
        ''' Try to find an unreachable state (after advancing the start state, if needed). Return True if anything happened. '''
        infinity = float('inf')
        did_something = False
        # Using Floyd-Warshall, calc LONGEST paths.
        max_dist = [[1 if x*y else 0 if x is y else -infinity for y in self.rewrites] for x in self.rewrites]
        for k, dk in enumerate(max_dist):
            for i, di in enumerate(max_dist):
                for j, dj in enumerate(max_dist):
                    di[j] = max(di[j], di[k] + dk[j])

        keep = None  # To be calculated: for each rewrite, whether one can reach it and loop from the starting state.
        # Before we delete all rules not satisfying this property, we must advance the starting state.
        while True:
            for root, rewrite in enumerate(self.rewrites):
                succ = rewrite.apply_to(self.start, as_tape=True)
                if succ is not self.start:
                    break
            else:
                # If the start word isn't rewritten, it's halting! Someone handed us a trivial case.
                did_something = bool(self.rewrites)
                for rewrite in self.rewrites:
                    self._add_word('unreachable', rewrite.f)
                self.rewrites.clear()
                break
            if keep is None:
                keep = [max_dist[root][i] > 0 and di[i] > 0 for i, di in enumerate(max_dist)]
            if keep[root]:
                break
            self.start = succ
            did_something = True

        for i, rewrite in enumerate(self.rewrites):
            if not keep[i]:
                did_something = True
                self._add_word('unreachable', rewrite.f)
        self.rewrites = [rewrite for i, rewrite in enumerate(self.rewrites) if keep[i]]

        for typ in 'halting', 'cycling':
            unreachable_special_words = [word for word in self.special_words[typ] if all(rewrite * Rewrite(word,word) is None for rewrite in self.rewrites)]
            for word in unreachable_special_words:
                self.special_words[typ].remove(word)
                self._add_word('unreachable', word)
                did_something = True

        return did_something

    def split_rule(self, rewrite, side, word):
        ''' Split "rewrite" (in self.rewrites) into equivalent disjoint rules which wholly do or don't match "word" on the given side.
            Return True if anything happened.'''
        ours = getattr(rewrite, side)
        l_len = min(len(word.l), len(ours.l))
        r_len = min(len(word.r), len(ours.r))
        core1 = word.slice(l_len, r_len)
        core2 = ours.slice(l_len, r_len)
        if core1 != core2 or (l_len, r_len) == (len(word.l), len(word.r)):
            return  False # "word" is disjoint from, or at least as broad as, this rewrite's domain.
        self.rewrites.remove(rewrite)
        while l_len < len(word.l):
            bit = word.l[-l_len]
            self._add_rule(Rewrite(flip(bit) + rewrite.f, flip(bit) + rewrite.t))
            rewrite = Rewrite(bit + rewrite.f, bit + rewrite.t)
            l_len += 1
        while r_len < len(word.r):
            bit = word.r[+r_len]
            self._add_rule(Rewrite(rewrite.f + flip(bit), rewrite.t + flip(bit)))
            rewrite = Rewrite(rewrite.f + bit, rewrite.t + bit)
            r_len += 1
        self._add_rule(rewrite)
        return True

    def split_rules(self, side, word):
        did_something = False
        while any(self.split_rule(rewrite, side, word) for rewrite in self.rewrites):
            did_something = True
            self.prune()

    def simplify_once(self):
        cyclic_states = {rewrite.f.s for rewrite in self.rewrites if rewrite.f.s == rewrite.t.s}
        rest = Counter(s for rewrite in self.rewrites for s in (rewrite.f.s, rewrite.t.s) if s not in cyclic_states)
        if not rest:
            return False
        s, _ = rest.most_common()[-1]
        domains = [rewrite.f for rewrite in self.rewrites if rewrite.f.s == s]
        codomains = [rewrite.t for rewrite in self.rewrites if rewrite.t.s == s]
        did_something = False
        while any(self.split_rule(rewrite, 't', domain) for rewrite in self.rewrites if rewrite.t.s == s for domain in domains):
            did_something = True
        while any(self.split_rule(rewrite, 'f', codomain) for rewrite in self.rewrites if rewrite.f.s == s for codomain in codomains):
            did_something = True
        return did_something

    def simplify(self):
        did_something = self.prune()
        while self.simplify_once():
            self.prune()
            did_something = True
        return did_something

    def simulate(self, steps):
        rules_used = []
        w = self.start
        for _ in range(steps):
            for i, rw in enumerate(self.rewrites):
                x = rw.apply_to(w, as_tape=True)
                if x is w: continue
                w = x
                rules_used.append(i)
                print(w)
                break
        delimiter = ',' if len(self.rewrites)>10 else ''
        print('Rules used:', delimiter.join(map(str, rules_used)))
        print('Frequency:', Counter(rules_used).most_common())

if __name__ == '__main__':
    from argparse import ArgumentParser
    ap = ArgumentParser(description='Try to simplify a TM as a string rewriting system.')
    ap.add_argument('--db', help='Path to DB file', type=str, default='all_5_states_undecided_machines_with_global_header')
    ap.add_argument('--simulate', help='Show this many steps.', type=int, default=0)
    ap.add_argument('--splitf', help='Word(s) to split domains ("from" side) on', type=str, nargs='*', default=[])
    ap.add_argument('--splitt', help='Word(s) to split codomains ("to" side) on', type=str, nargs='*', default=[])
    ap.add_argument('seeds', help='DB seed numbers', type=int, nargs='*', default=[7410754, 5878998])
    args = ap.parse_args()

    for seed in args.seeds:
        print('='*40, seed, '='*40)
        machine = get_machine_i(args.db, seed)
        S = RewriteSystem(machine)
        for word_str in args.splitf:
            S.split_rules('f', Word.from_str(word_str))
        for word_str in args.splitt:
            S.split_rules('t', Word.from_str(word_str))
        S.simplify()
        if (args.simulate):
            S.simulate(steps=args.simulate)
        print(S)
