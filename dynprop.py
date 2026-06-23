import os
import argparse
import numpy as np


def dyn_prop(id, lines):
    prev_dist = 1.0
    dyn_props = []
    for l1 in lines:
        l = l1.split(',')
        if l[1] == id:
            box_prop = float(l[2]) / prev_dist

            prev_dist = float(l[2])
            dyn_props.append([l[0], id, box_prop])

    return dyn_props


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--static_path",type=str,required=True)

    args = parser.parse_args()

    static_path = args.static_path
    static = open(static_path, 'r')

    header = static.readline()
    s_lines = static.readlines()
    # print(s_lines)

    file = open(static_path.replace("sp.csv", "dyn_prop.csv"), 'w')
    file.write("Frame,ID,Dynamic Proportion\n")
    

    output = set()
    for x1 in s_lines:
        x = x1.split(',')
        output.add(x[1])
    

    for id in output:
        d_list = dyn_prop(id, s_lines)
        for d in d_list:
            file.write(d[0] + ',' + d[1] + ',' + str(d[2]) + '\n')