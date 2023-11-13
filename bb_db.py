# SPDX-FileCopyrightText: 2023 Justin Blanchard <UncombedCoconut@gmail.com>
# SPDX-License-Identifier: Apache-2.0 OR MIT
from bbchallenge import get_header
from bb_tm import TM
from collections.abc import Sequence
from functools import cached_property
import os

class DB(Sequence):
    def __init__(self, path, states, symbols):
        self._path, self._states, self._symbols = path, states, symbols
        self._file = None
        self._entered = 0

    def __len__(self):
        return int.from_bytes(get_header(self._path)[8:12], byteorder='big')

    def __enter__(self):
        if self._entered <= 0:
            self._file = open(self._path, 'rb')
        self._entered += 1

    def __exit__(self, *exc_info):
        self._entered -= 1
        if self._entered <= 0:
            self._file.close()
            self._file = None

    def __getitem__(self, seed):
        with self:
            self._file.seek(30 + seed * self._tm_size)
            return TM(self._file.read(self._tm_size), self._states, self._symbols, seed)

    def __iter__(self):
        with self:
            seed = 0
            self._file.seek(30)
            while (code := self._file.read(self._tm_size)):
                yield TM(code, self._states, self._symbols, seed)
                seed += 1

    @cached_property
    def _tm_size(self):
        return 3*self._states*self._symbols

class Index(list):
    def __init__(self, path=None):
        super().__init__()
        if path:
            with open(path, 'rb') as f:
                while (u32 := f.read(4)):
                    self.append(int.from_bytes(u32, 'big'))

    def save(path):
        with open(path, 'wb') as f:
            for seed in self:
                f.write(seed.to_bytes(4, 'big'))
