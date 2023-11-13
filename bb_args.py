# SPDX-FileCopyrightText: 2023 Justin Blanchard <UncombedCoconut@gmail.com>
# SPDX-License-Identifier: Apache-2.0 OR MIT
from argparse import Action, ArgumentParser
from bb_db import DB, Index, TM
from functools import cached_property

def tm_args(nargs='*'):
    """Return an ArgumentParser that lets the user specify a DB and seeds (for instance), parsed into 'machines': Collection[TuringMachine]. """
    ap = ArgumentParser(add_help=False)
    ap.add_argument('-d', '--db', help='Path to DB file', default='all_5_states_undecided_machines_with_global_header')
    ap.add_argument('-n', '--states', help='Number of TM states (for seed-DB TMs)', type=int, default=5)
    ap.add_argument('-s', '--symbols', help='Number of tape symbols (for seed-DB TMs)', type=int, default=2)
    ap.add_argument('-i', '--index', help='Include all machines from this index file')
    ap.add_argument('machines', help='DB seed numbers', nargs=nargs, action=_TMAction)
    return ap

class _TMAction(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, BeaverColony(namespace, values))

class BeaverColony:
    def __init__(self, namespace, values):
        self._namespace, self._values = namespace, values

    @cached_property
    def _index(self):
        return Index(self._namespace.index)

    @cached_property
    def _db(self):
        return DB(self._namespace.db, self._namespace.states, self._namespace.symbols)

    def __len__(self):
        return len(self._index) + len(self._values) or len(self._db)

    def __iter__(self):
        with self._db:
            for seed_or_text in self._values:
                try:
                    seed = int(seed_or_text)
                    yield self._db[seed]
                except ValueError:
                    yield TM.from_text(seed_or_text)
            for seed in self._index:
                yield self._db[seed]
            if not (self._index or self._values):
                yield from self._db
