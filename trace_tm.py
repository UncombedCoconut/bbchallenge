#!/usr/bin/env pypy3
# SPDX-FileCopyrightText: 2025 Justin Blanchard <UncombedCoconut@gmail.com>
# SPDX-License-Identifier: Apache-2.0 OR MIT
STATE_COLOR = ((255, 0, 0), (255, 128, 0), (0, 0, 255), (0, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0))

def main(tm, step_limit, tape_l=0, tape_r=0, png=False, svg=False):
    name = f'{tm.states}x{tm.symbols}_{tm.seed}' if tm.seed else str(tm)
    fn = f'bb_{name}.txt'
    with open(fn, 'w') as bbout:
        curr_state = 0
        curr_pos = step_limit
        tape = bytearray(2*step_limit)
        l = curr_pos + tape_l
        r = curr_pos + tape_r
        for row in range(step_limit):
            l = min(l, curr_pos)
            r = max(r, curr_pos)
            print(*tape[l:curr_pos], sep='', end='', file=bbout)
            print(f'{chr(65+curr_state)}', end='', file=bbout)
            print(*tape[curr_pos:r+1], sep='', file=bbout)
            assert curr_state >= 0
            w, d, t = tm.transition(curr_state, tape[curr_pos])
            tape[curr_pos] = w
            curr_pos += (-1 if d else 1)
            curr_state = t
    if png: save_png(tm, step_limit, name, l, r)
    if svg: save_svg(tm, step_limit, name, l, r)

def save_png(tm, step_limit, name, l, r):
    from PIL import Image
    l, curr_pos, r = 0, step_limit - l, r - l
    curr_state = 0
    tape = bytearray(r+1)
    img = Image.new('RGB', (r+1, step_limit), color='black')
    pix = img.load()
    for row in range(step_limit):
        for i, s in enumerate(tape):
            pix[i, row] = (round(255 * s / (tm.symbols-1)),) * 3
        pix[curr_pos, row] = STATE_COLOR[curr_state]
        assert curr_state >= 0
        w, d, t = tm.transition(curr_state, tape[curr_pos])
        tape[curr_pos] = w
        curr_pos += (-1 if d else 1)
        curr_state = t
    img.save(f'bb_{name}.png')

def save_svg(tm, step_limit, name, l, r):
    from drawsvg import Drawing, Group, Rectangle
    SVG_CELL_SIZE = 32

    l, curr_pos, r = 0, step_limit - l, r - l
    curr_state = 0
    tape = bytearray(r+1)
    img = Drawing(SVG_CELL_SIZE*(r+1), SVG_CELL_SIZE*step_limit, displayInline=False)
    main_group = Group()
    main_group.append(Rectangle(0, 0, img.width, img.height, fill='black'))
    for row in range(step_limit):
        row_group = Group()
        for i, s in enumerate(tape):
            color = STATE_COLOR[curr_state] if i == curr_pos else (round(255 * s / (tm.symbols-1)),) * 3
            fill = 'rgb({},{},{})'.format(*color)
            pos_x = SVG_CELL_SIZE*i
            pos_y = SVG_CELL_SIZE*row
            row_group.append(Rectangle(pos_x, pos_y, SVG_CELL_SIZE, SVG_CELL_SIZE, fill=fill))
        main_group.append(row_group)
        assert curr_state >= 0
        w, d, t = tm.transition(curr_state, tape[curr_pos])
        tape[curr_pos] = w
        curr_pos += (-1 if d else 1)
        curr_state = t
    img.append(main_group)
    img.save_svg(f'bb_{name}.svg')

if __name__ == '__main__':
    from bb_args import ArgumentParser, tm_args
    ap = ArgumentParser(description='Diagram a TM (text/png/svg).', parents=[tm_args()])
    ap.add_argument('-N', '--step-limit', help='Number of rows to show.', type=int, default=10000)
    ap.add_argument('-l', '--tape-left', help='Tape includes this x position', type=int, default=0)
    ap.add_argument('-r', '--tape-right', help='Tape includes this x position', type=int, default=0)
    ap.add_argument('-p', '--png', help='Emit a PNG file', action='store_true')
    ap.add_argument('-v', '--svg', help='Emit an SVG file', action='store_true')
    args = ap.parse_args()
    for tm in args.machines:
        main(tm, step_limit=args.step_limit, tape_l=args.tape_left, tape_r=args.tape_right, png=args.png, svg=args.svg)
