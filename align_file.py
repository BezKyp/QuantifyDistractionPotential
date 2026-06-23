import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",type=str,required=True)

    args = parser.parse_args()

    file = args.file
    f = open(file, 'r')
    f_lines = f.readlines()
    l0 = f_lines[-1].split(",")

    frames = int(l0[0])
    print("frames: " + str(frames))

    new_file = open(file.replace(".csv", "_new.csv"), 'w')
    new_file.write("Frame,Side,X Coord,Y Coord\n")

    f = 0
    for i in range (1, frames):
        print("i: " + str(i))
        l = f_lines[i].split(",")
        if l[0] == frames:
            new_file.write(f_lines[i])
            break
        
        if f < int(l[0]):
            for x in range(f, int(l[0])):
                new_file.write(str(x) + ",Neither,-1,-1\n")

            new_file.write(f_lines[i])
            f = int(l[0])
        else:
            new_file.write(f_lines[i])
        
        f += 1
