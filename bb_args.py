# SPDX-FileCopyrightText: 2023 Justin Blanchard <UncombedCoconut@gmail.com>
# SPDX-License-Identifier: Apache-2.0 OR MIT
from argparse import Action, ArgumentParser
from bb_db import DB, Index, TM
from collections.abc import Sequence
from functools import cached_property

def tm_args():
    """Return an ArgumentParser that lets the user specify a DB and seeds (for instance), parsed into 'machines': Sequence[TuringMachine]. """
    ap = ArgumentParser(add_help=False)
    ap.add_argument('-d', '--db', help='Path to DB file', default=DB.DEFAULT_PATH)
    ap.add_argument('-n', '--states', help='Number of TM states (for seed-DB TMs)', type=int, default=5)
    ap.add_argument('-s', '--symbols', help='Number of tape symbols (for seed-DB TMs)', type=int, default=2)
    ap.add_argument('-i', '--index', help='Include all machines from this index file', type=Index, action=_AddMachineList)
    ap.add_argument('machines', help='Standard text TMs or DB seed numbers', nargs='*', action=_AddMachineList, default=BeaverColony())
    return ap

class _AddMachineList(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        namespace.machines._namespace = namespace
        if values and values is not self.default:
            namespace.machines._lists.append(values)

class BeaverColony(Sequence):
    def __init__(self):
        self._namespace = None
        self._lists = []

    @cached_property
    def _db(self):
        return DB(self._namespace.db, self._namespace.states, self._namespace.symbols)

    def __len__(self):
        return sum(map(len, self._lists or [self._db]))

    def __getitem__(self, i):
        for l in self._lists or [self._db]:
            if i < len(l):
                return self._tm(l[i])
            i -= len(l)
        raise IndexError('list index out of range')

    def __iter__(self):
        with self._db:
            for l in self._lists or [self._db]:
                for seed_or_text in l:
                    yield self._tm(seed_or_text)

    def _tm(self, seed_or_text):
        if isinstance(seed_or_text, TM):
            return seed_or_text
        try:
            seed = int(seed_or_text)
            return self._db[seed]
        except ValueError:
            return TM.from_text(seed_or_text)
