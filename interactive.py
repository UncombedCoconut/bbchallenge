#!/usr/bin/pypy3
# SPDX-FileCopyrightText: 2022 Justin Blanchard <UncombedCoconut@gmail.com>
# SPDX-License-Identifier: Apache-2.0 OR MIT
from copy import deepcopy
from dataclasses import dataclass
from functools import partial

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk
from gi.repository import Gdk
from xdot import DotWidget, DotWindow
from xdot.ui.elements import Edge, Node

from string_rewrite import get_machine_i, Rewrite, RewriteSystem, Word


class GUI(DotWidget):
    def __init__(self, srs):
        super().__init__()
        self.srs = deepcopy(srs)
        self.srs.prune()
        self.undo_stack = []
        self.refresh()

    def refresh(self):
        self.set_dotcode('\n'.join(self.dot_lines()).encode())
        return True

    def on_key_press_event(self, widget, event):
        if event.keyval == Gdk.KEY_s:
            return self.action('simplify')
        elif event.keyval == Gdk.KEY_z:
            return self.on_undo()
        return super().on_key_press_event(widget, event)

    def on_click(self, element, event):
        if isinstance(element, Edge):
            if event.button == 1:
                rw, restricted = self.edges[element.src.id, element.dst.id]
                return self.action('split_rule', rw, 'f', restricted.f)
            else:
                rw, restricted = self.edges[element.src.id, element.dst.id]
                return self.action('split_rule', rw, 't', restricted.t)
        elif isinstance(element, Node):
            if event.button == 1:
                return self.action('split_rules', 'f', '0'+self.nodes[element.id])
            else:
                return self.action('split_rules', 'f', self.nodes[element.id]+'0')
        return super().on_click(element, event)

    def dot_lines(self):
        self.nodes = {} # id -> Word
        self.edges = {} # (src_id, dst_id) -> (rewrite, restricted_rewrite)
        yield 'digraph G {'
        for i, cat in enumerate(('halting', 'cycling')):
            if self.srs.special_words[cat]:
                for w in self.srs.special_words[cat]:
                    self.nodes[str(w).encode()] = w
                    yield f'"{str(w)}" [peripheries={i+2}]'
        self.nodes.update((str(rw.f).encode(), rw.f) for rw in self.srs.rewrites)
        for rw in self.srs.rewrites:
            for w in self.nodes.values():
                restricted = rw.then(Rewrite(w, w))
                if restricted is not None:
                    self.edges[str(rw.f).encode(), str(w).encode()] = (rw, restricted)
                    yield f'"{str(rw.f)}" -> "{str(w)}" [label="{str(rw)}"]'
        yield '}'

    def action(self, method, *args):
        self.undo_stack.append((deepcopy(self.srs), f'{method}{args}'))
        print('ACTION', self.undo_stack[-1][1])
        getattr(self.srs, method)(*args)
        self.srs.prune()
        print(self.srs)
        print()
        return self.refresh()

    def on_undo(self):
        if self.undo_stack:
            self.srs, action = self.undo_stack.pop()
            print('UNDO', action)
            return self.refresh()


if __name__ == '__main__':
    from argparse import ArgumentParser
    ap = ArgumentParser(description='Try to simplify a TM as a string rewriting system.')
    ap.add_argument('-d', '--db', help='Path to DB file', default='all_5_states_undecided_machines_with_global_header')
    ap.add_argument('seeds', help='DB seed numbers', type=int, nargs='+')
    args = ap.parse_args()

    for seed in args.seeds:
        machine = get_machine_i(args.db, seed)
        s = RewriteSystem(machine)
        w = GUI(s)
        DotWindow(widget=w).connect('delete-event', Gtk.main_quit)
    Gtk.main()
