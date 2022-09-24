# SPDX-FileCopyrightText: 2022 Tristan Stérin <tristan.sterin@mu.ie>
# SPDX-License-Identifier: MIT
# This is simply an importable form of the utility functions in https://github.com/bbchallenge/bbchallenge-py/blob/main/BB5.ipynb
import os

DB_PATH = "all_5_states_undecided_machines_with_global_header"

def get_header(machine_db_path):
    with open(machine_db_path, "rb") as f:
        return f.read(30)

header = get_header(DB_PATH)
undecided_time, undecided_space, undecided_total = int.from_bytes(header[:4],byteorder='big'),int.from_bytes(header[4:8],byteorder='big'),int.from_bytes(header[8:12],byteorder='big')
#print(undecided_time, undecided_space, undecided_total, header[12])
#n = os.path.getsize(DB_PATH)

#print((n)/30-1)

def get_machine_i(machine_db_path, i, db_has_header=True):
    with open(machine_db_path, "rb") as f:
        c = 1 if db_has_header else 0
        f.seek(30*(i+c))
        return f.read(30)

def ithl(i):
    return chr(ord("A")+i)

def g(move):
    if move == 0:
        return "R"
    return "L"

def pptm(machine, return_repr=False):
    from tabulate import tabulate
    headers = ["s","0","1"]
    table = []
    
    for i in range(5):
        row = [ithl(i)]
        for j in range(2):
            write = machine[6*i+3*j] 
            move = machine[6*i+3*j+1] 
            goto = machine[6*i+3*j+2]-1
            
            if goto == -1:
                row.append("???")
                continue
                
            row.append(f"{write}{g(move)}{ithl(goto)}")
        table.append(row)
    
    if not return_repr:
        print(tabulate(table,headers=headers))
    else:
        return tabulate(table,headers=headers)

#pptm(get_machine_i(DB_PATH,0,db_has_header=True))
#pptm(get_machine_i(DB_PATH,10,db_has_header=True))
#pptm(get_machine_i(DB_PATH,4888230,db_has_header=True))

def step(machine, curr_state, curr_pos, tape):
    if not curr_pos in tape:
        tape[curr_pos] = 0
    
    write = machine[curr_state*6 + 3*tape[curr_pos]] 
    move = machine[curr_state*6 + 3*tape[curr_pos] + 1] 
    goto = machine[curr_state*6 + 3*tape[curr_pos] + 2] - 1

    if goto == -1:
        return None, None
    
    tape[curr_pos] = write
    next_pos = curr_pos  + (-1 if move else 1)
    return goto, next_pos

def simulate(machine, time_limit = 1000, mini =-10, maxi=-10):
    curr_time = 0
    curr_state = 0
    curr_pos = 0
    tape = {}
    
    while curr_state != None and curr_time < time_limit:
        curr_state, curr_pos = step(machine, curr_state, curr_pos, tape)
        if curr_state is not None:
            pprinttape(tape, curr_state, curr_pos, mini=mini, maxi=maxi)
        else:
            print("HALT")
        curr_time += 1
        
def tm_trace_to_image(machine, width=900, height=1000, origin=0.5, show_head_direction=False):
    from PIL import Image
    img = Image.new('RGB', (width, height), color = 'black')
    pixels = img.load()
    
    
    tape = {}
    curr_time = 0
    curr_state = 0
    curr_pos = 0
    tape = {}
    
    
    for row in range(1,height):
        last_pos = curr_pos
        curr_state, curr_pos = step(machine, curr_state, curr_pos, tape)
        
        if curr_state is None: #halt
            return img
        
        for col in range(width):
            pos = col-width*(origin)
            
            if pos in tape:
                pixels[col,row] = (255,255,255) if tape[pos] == 1 else (0,0,0)
                #pixels[col,row-1] = colors[curr_state-1]
                
            if pos == curr_pos and show_head_direction:
                pixels[col,row] = (255,0,0) if curr_pos > last_pos else (0,255,0) 
                
                
    #img = zoom_at(img,*zoom)
    return img

def zoom_at(img, x, y, zoom):
    from PIL import Image
    w, h = img.size
    zoom2 = zoom * 2
    img = img.crop((x - w / zoom2, y - h / zoom2, 
                    x + w / zoom2, y + h / zoom2))
    return img.resize((w, h), Image.LANCZOS)

def repr_to_bytes(rep):
    to_rep = bytearray()
    for a in rep:
        to_rep.append(a)
    return to_rep
