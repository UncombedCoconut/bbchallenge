#!/usr/bin/python3
# SPDX-FileCopyrightText: 2022 Justin Blanchard <UncombedCoconut@gmail.com>
# SPDX-License-Identifier: MIT
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
import matplotlib.pyplot as plt
import networkx as nx
from string_rewrite import get_machine_i, DB_PATH, Rewrite, RewriteSystem, Word


class GUI:
    def __init__(self, srs):
        self.srs = srs
        self.undo_stack = []
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.mpl_connect('key_press_event', self.key_down)
        self.keymap = {'e': self.cb_lsplit, 'E': self.cb_rsplit, 'f': self.cb_fsplit, 't': self.cb_tsplit, 'p': self.cb_prune, 's': self.cb_simplify, 'z': self.cb_undo}
        self.refresh()

    def refresh(self):
        self.ax.clear()
        g = self.graph()
        pos = nx.drawing.kamada_kawai_layout(g)
        self.nx_nodes = nx.draw_networkx_nodes(g, pos, ax=self.ax)
        self.nx_edges = nx.draw_networkx_edges(g, pos, ax=self.ax, arrowsize=50)
        nx.draw_networkx_labels(g, pos, ax=self.ax)
        self.fig.canvas.draw_idle()

    def key_down(self, event):
        sel_edge = sel_node = None
        if event.inaxes is self.ax:
            for i, e in enumerate(self.nx_edges):
                if e.contains(event)[0]:
                    sel_edge = i
            cont, ind = self.nx_nodes.contains(event)
            if cont:
                sel_node = ind['ind'][0]
        f = self.keymap.get(event.key)
        if f:
            f(sel_node, sel_edge)
            self.refresh()

    def graph(self):
        g = nx.DiGraph()
        self.nodes = [] # (word, label)
        self.edges = [] # (rewrite, restricted, to, label)
        for cat in 'halting', 'cycling':
            if self.srs.special_words[cat]:
                for w in self.srs.special_words[cat]:
                    self.nodes.append((w, f'{cat[0].upper()}: {str(w)}'))
        self.nodes.extend((rw.f, str(rw.f)) for rw in self.srs.rewrites)
        g.add_nodes_from(t[1] for t in self.nodes)
        for rw in self.srs.rewrites:
            for w, label in self.nodes:
                restricted = rw.then(Rewrite(w, w))
                if restricted is not None:
                    self.edges.append((rw, restricted, w, str(rw)))
                    g.add_edge(str(rw.f), label)
        return g

    def cb_lsplit(self, sel_node, sel_edge):
        if sel_edge is not None:
            w = self.edges[sel_edge][1].f
        elif sel_node is not None:
            w = self.nodes[sel_node][0]
        else:
            return
        self.action('split_rules', 'f', '0'+w)

    def cb_rsplit(self, sel_node, sel_edge):
        if sel_edge is not None:
            w = self.edges[sel_edge][1].f
        elif sel_node is not None:
            w = self.nodes[sel_node][0]
        else:
            return
        self.action('split_rules', 'f', w+'0')

    def cb_fsplit(self, sel_node, sel_edge):
        if sel_edge is not None:
            rw, restricted = self.edges[sel_edge][:2]
            self.action('split_rule', rw, 'f', restricted.f)

    def cb_tsplit(self, sel_node, sel_edge):
        if sel_edge is not None:
            rw, restricted = self.edges[sel_edge][:2]
            self.action('split_rule', rw, 't', restricted.t)

    def cb_prune(self, sel_node, sel_edge):
        self.action('prune')

    def cb_simplify(self, sel_node, sel_edge):
        self.action('simplify')

    def action(self, method, *args):
        self.undo_stack.append((deepcopy(self.srs), f'{method}{args}'))
        print('ACTION', self.undo_stack[-1][1])
        getattr(self.srs, method)(*args)
        print(self.srs)
        print()
        self.refresh()

    def cb_undo(self, sel_node, sel_edge):
        if self.undo_stack:
            self.srs, action = self.undo_stack.pop()
            print('UNDO', action)
            self.refresh()


if __name__ == '__main__':
    from argparse import ArgumentParser
    ap = ArgumentParser(description='Try to simplify a TM as a string rewriting system.')
    ap.add_argument('--db', help='Path to DB file', type=str, default=DB_PATH)
    ap.add_argument('--splitf', help='Word(s) to split domains ("from" side) on', type=str, nargs='*', default=[])
    ap.add_argument('--splitt', help='Word(s) to split codomains ("to" side) on', type=str, nargs='*', default=[])
    ap.add_argument('--seed', help='DB seed number', type=int, default=7410754)
    args = ap.parse_args()

    machine = get_machine_i(args.db, args.seed)
    S = RewriteSystem(machine)
        #S.split_rules('f', Word.from_str(word_str))
        #S.split_rules('t', Word.from_str(word_str))
    for k, v in plt.rcParams.items():
        if k.startswith('keymap.') and k != 'keymap.quit':
            v.clear()
    GUI(S)
    plt.show()
