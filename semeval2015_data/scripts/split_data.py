import sys
import argparse


parser = argparse.ArgumentParser(
    'split SemEval 2014/2015 files into train/dev/test.')
parser.add_argument('FILE', help='SemEval file to split')
parser.add_argument('IDS', help='ids of sentences to be contained')
parser.add_argument('--sem2014', action='store_true', help='ids of sentences to be contained')
args = parser.parse_args()

trim_first_line = not args.sem2014  # True for SemEval 2015, False for SemEval 2014.

with open(args.IDS) as f:
    docs = {line.rstrip('\n'): False for line in f}

with open(args.FILE) as f:
    selected = False

    if trim_first_line:
        line = f.readline()
        print(line.rstrip('\n'))

    for line in f:
        line = line.rstrip('\n')
        if line.startswith('#') and line.split('\t')[0] != '#':
            if line in docs:
                selected = docs[line] = True
                print(line)
            else:
                selected = False
        elif selected:
            print(line)

for line in docs:
    if not docs[line]:
        print(f'Document not found: {line}', file=sys.stderr)
