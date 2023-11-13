#!/usr/bin/pypy3
# SPDX-FileCopyrightText: 2023 Justin Blanchard <UncombedCoconut@gmail.com>
# SPDX-License-Identifier: Apache-2.0 OR MIT
from argparse import Action, ArgumentParser
from bbchallenge import get_header, get_machine_i
from bb_tm import TM
from functools import cached_property

def tm_args(nargs='*'):
    """Return an ArgumentParser that lets the user specify a DB and seeds (for instance), parsed into 'machines': Collection[TuringMachine]. """
    ap = ArgumentParser(add_help=False)
    ap.add_argument('-d', '--db', help='Path to DB file', default='all_5_states_undecided_machines_with_global_header')
    ap.add_argument('-n', '--states', help='Number of TM states (for seed-DB TMs)', type=int, default=5)
    ap.add_argument('-s', '--symbols', help='Number of tape symbols (for seed-DB TMs)', type=int, default=2)
    ap.add_argument('machines', help='DB seed numbers', nargs=nargs, action=_TMAction)
    return ap

class _TMAction(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, BeaverColony(namespace, values))

class BeaverColony:
    def __init__(self, namespace, values):
        self._namespace = namespace
        self._values = values

    @cached_property
    def seeds(self):
        return [int(v) for v in self._values] or range(int.from_bytes(get_header(self._namespace.db)[8:12], byteorder='big'))

    def __iter__(self):
        with open(self._namespace.db, 'rb') as db:
            header_size, tm_size = 30, 3*self._namespace.states*self._namespace.symbols
            for seed in self.seeds:
                db.seek(header_size + seed * tm_size)
                yield TM(db.read(tm_size), self._namespace.states, self._namespace.symbols, seed)
