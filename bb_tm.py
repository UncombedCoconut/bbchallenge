#!/usr/bin/pypy3
# SPDX-FileCopyrightText: 2023 Justin Blanchard <UncombedCoconut@gmail.com>
# SPDX-License-Identifier: Apache-2.0 OR MIT

class TM:
    __slots__ = ('code', 'states', 'symbols', 'seed')
    def __init__(self, code, states=5, symbols=2, seed=None):
        self.code, self.states, self.symbols, self.seed = code, states, symbols, seed

    def transition(self, from_state, read_symbol):
        """ Return (write, direction, to_state). (directions are R=0, L=1; to_state==-1 is halt.) """
        fr = from_state * self.symbols + read_symbol
        w, d, t = self.code[3*fr:3*(fr+1)]
        return w, d, t-1

    def transitions(self):
        """ Yield tuples (from_state, read, write, direction, to_state). (directions are R=0, L=1; to_state==-1 is halt.) """
        fr_x_3 = 0
        for f in range(self.states):
            for r in range(self.symbols):
                w, d, t = self.code[fr_x_3:fr_x_3+3]
                yield f, r, w, d, t-1
                fr_x_3 += 3

    def __str__(self):
        parts = []
        for f, r, w, d, t in self.transitions():
            if f > 0 and r == 0:
                parts.append('_')
            parts.append('---' if t < 0 else f'{w}{"RL"[d]}{chr(65+t)}')
        return ''.join(parts)

    def __reversed__(self):
        mirror_tm = bytearray(self.code)
        for i in range(0, len(mirror_tm), 3):
            mirror_tm[i+1] ^= 1
        return type(self)(mirror_tm, self.states, self.symbols)
