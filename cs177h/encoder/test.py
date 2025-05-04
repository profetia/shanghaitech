import sys
import joblib


filename = sys.argv[1]

def process_one_line(line):
    line[-2] = "\t"

with open(filename, "r") as f:
    lines = f.read().split("\n")

lines = [lines[0], *joblib.Parallel(n_jobs=-1)(joblib.delayed(process_one_line)(line) for line in lines[1:])]

with open(f"{filename}_new.txt", "w") as f:
    f.write("\n".join(lines))

    