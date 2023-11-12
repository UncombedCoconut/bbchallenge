#!/usr/bin/pypy3
# SPDX-FileCopyrightText: 2022 Justin Blanchard <UncombedCoconut@gmail.com>
# SPDX-License-Identifier: Apache-2.0 OR MIT
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass
from functools import partial

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk
from gi.repository import Gdk
from xdot import DotWidget, DotWindow
from xdot.ui.elements import Edge, Node

from string_rewrite import Rewrite, RewriteSystem, Word


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
        if event.keyval == Gdk.KEY_w:
            return self.action('sandwich')
        elif event.keyval == Gdk.KEY_f:
            return self.action('split_first_cycle')
        elif event.keyval == Gdk.KEY_a:
            return self.action('advance_start')
        elif event.keyval == Gdk.KEY_2:
            return self.action('exponentiate', 2)
        elif event.keyval == Gdk.KEY_3:
            return self.action('exponentiate', 3)
        elif event.keyval == Gdk.KEY_5:
            return self.action('exponentiate', 5)
        elif event.keyval == Gdk.KEY_z:
            return self.on_undo()
        elif event.keyval == Gdk.KEY_i:
            self.srs.simulate(4*len(self.srs.rewrites))
            return True
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
        _, i0 = self.srs.step(self.srs.start)
        yield 'digraph G {'
        yield f'"{str(self.srs.rewrites[i0].f)}" [shape=diamond]'
        for i, cat in enumerate(('halting', 'cycling')):
            if self.srs.special_words[cat]:
                for w in self.srs.special_words[cat]:
                    self.nodes[str(w).encode()] = w
                    yield f'"{str(w)}" [peripheries={i+2}]'
        self.nodes.update((str(rw.f).encode(), rw.f) for rw in self.srs.rewrites)
        adj_list = defaultdict(list)
        for rw in self.srs.rewrites:
            for w in self.nodes.values():
                restricted = rw.then(Rewrite(w, w))
                if restricted is not None:
                    b_rw_f, b_w = str(rw.f).encode(), str(w).encode()
                    self.edges[b_rw_f, b_w] = (rw, restricted)
                    adj_list[b_rw_f].append(b_w)

        # DFS, for coloring
        q = deque([str(self.srs.rewrites[i0].f).encode()])
        clock, t0, t1, parent = 0, {}, {}, {}
        while q:
            v = q[-1]
            al = adj_list[v]
            if v not in t0:
                t0[v] = clock
                al.reverse()
            clock += 1
            if al:
                child = al.pop()
                if child not in t0:
                    parent[child] = v
                    q.append(child)
            else:
                t1[v] = clock
                q.pop()
            clock += 1
        COLOR = {(False, True): 'red', (True, False): 'green', (False, False): 'blue'}  # Back, forward, cross

        for (b_rw_f, b_w), (rw, restricted) in self.edges.items():
            color = '' if parent.get(b_w) == b_rw_f else f', color={COLOR[t0.get(b_rw_f)<t0.get(b_w), t1.get(b_rw_f)<t1.get(b_w)]}'
            yield f'"{str(rw.f)}" -> "{b_w.decode()}" [label="{str(rw)}"{color}]'
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
    from bb_args import ArgumentParser, tm_args
    ap = ArgumentParser(description='Try to simplify a TM as a string rewriting system.', parents=[tm_args()])
    args = ap.parse_args()

    for tm in args.machines:
        s = RewriteSystem(tm)
        w = GUI(s)
        DotWindow(widget=w).connect('delete-event', Gtk.main_quit)
    Gtk.main()
