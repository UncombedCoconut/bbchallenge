#!/usr/bin/pypy3
# SPDX-FileCopyrightText: 2023 Justin Blanchard <UncombedCoconut@gmail.com>
# SPDX-License-Identifier: Apache-2.0 OR MIT
HALT, R, L = range(-1, 2)

class TM:
    __slots__ = ('code', 'states', 'symbols', 'seed')
    def __init__(self, code, states=5, symbols=2, seed=None):
        self.code, self.states, self.symbols, self.seed = code, states, symbols, seed

    def transition(self, from_state, read_symbol):
        """ Return (write, direction, to_state). (directions are R or L; to_state is HALT or a 0-based ID.) """
        fr = from_state * self.symbols + read_symbol
        w, d, t = self.code[3*fr:3*(fr+1)]
        return w, d, t-1

    def transitions(self):
        """ Yield tuples (from_state, read, write, direction, to_state). (directions are R or L; to_state is HALT or a 0-based ID.) """
        fr_x_3 = 0
        for f in range(self.states):
            for r in range(self.symbols):
                w, d, t = self.code[fr_x_3:fr_x_3+3]
                yield f, r, w, d, t-1
                fr_x_3 += 3

    def default_state(self):
        return 0

    def default_symbol(self):
        return 0

    def base_symbol(self, s):
        return s

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

    @classmethod
    def from_text(cls, text):
        tt_rows = text.split('_')
        N, S = len(tt_rows), len(tt_rows[0])//3
        if not all(len(row) == 3*S for row in tt_rows):
            raise ValueError(f'Not in standard TM text format: {text!r}')
        code = bytearray(3*N*S)
        for f, row in enumerate(tt_rows):
            for r, (w, d, t) in enumerate(zip(row[::3], row[1::3], row[2::3])):
                code[3*(f*S+r):3*(f*S+r+1)] = (0,0,0) if t=='-' else (int(w), 'RL'.index(d), ord(t)-64)
        return cls(bytes(code), N, S)
