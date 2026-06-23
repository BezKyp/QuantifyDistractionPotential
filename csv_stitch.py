import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir",type=str,required=True)

    args = parser.parse_args()

    basedir = args.dir
    dir_list = os.listdir(basedir)
    dir_list.sort()

    full_file = open(os.path.join(basedir, "final.csv"), 'w')

    i = 0
    for d in dir_list:
        file = open(os.path.join(basedir, d), 'r')
        header = file.readline()
        full = file.readlines()

        if i == 0:
            full_file.write(header)
            for f in full:
                full_file.write(f)
        else:
            for f in full:
                full_file.write(f)
        
        i += 1
        
    
