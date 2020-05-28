import sys
import argparse

parser = argparse.ArgumentParser(
    'join SemEval 2014/2015 files and its companion file.\n',
    'companion file is optional and contains a dependency tree annotation ',
    'in CoNLL like two-columned format')
parser.add_argument('FILE', help='SemEval file to split')
parser.add_argument('COMPANION', nargs='?', help='ids of sentences to be contained')
parser.add_argument('--sem2014', action='store_true', help='ids of sentences to be contained')
args = parser.parse_args()

has_sense = not args.sem2014  # True for SemEval 2015, False for SemEval 2014.
trim_first_line = not args.sem2014  # True for SemEval 2015, False for SemEval 2014.


def read_companion():
    if args.COMPANION:
        with open(args.COMPANION) as f:
            for line in f:
                yield list(line.rstrip('\n').split('\t'))[:3]
    else:
        while True:
            yield '_', '_', '_'

with open(args.FILE) as f:
    if trim_first_line:
        line = f.readline()
        print(line.rstrip('\n'))

    for line, comp_line in zip(f, read_companion()):
        line = line.rstrip('\n')
        if line == '' or (
            line.startswith('#') and
            line.split('\t')[0] != '#'
        ):
            print(line)
        else:
            fields = line.split('\t')
            word_index, word, lemma, pos, *_ = fields
            if len(fields) == 4:
                top, pred = '-', '-'
                if has_sense:
                    sense = '-'
            else:
                top, pred = fields[4], fields[5]
                if has_sense:
                    sense = fields[6]
            arg = fields[7:] if has_sense else fields[6:]
            predicted_pos, head, deprel = comp_line

            fields_output = [
                word_index,
                word,
                lemma,
                pos, #predicted_pos
                head,
                deprel,
                top,
                pred
            ]
            if has_sense:
                fields_output.append(sense);
            fields_output.extend(arg);
            print('\t'.join(fields_output))

