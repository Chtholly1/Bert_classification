#

import sys

talk_set = set()
def load_pre_val(file_path):
    with open(file_path) as f:
        for line in f:
            talk, stand, label = line.strip().split('\t')
            talk_set.add(talk)

load_pre_val('./data/talk_to_stand.val_13.5k')
f1 = open(sys.argv[1], 'w')
f2 = open(sys.argv[2], 'w')

for line in sys.stdin:
    talk, stand, label = line.strip().split('\t')
    if talk in talk_set:
        f2.write(line)
    else:
        f1.write(line)
f1.close()
f2.close()
