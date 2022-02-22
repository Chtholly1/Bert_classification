#coding:utf-8
import sys
import collections
from hanziconv import HanziConv
from transformers import BertTokenizer

MODEl_NAME = './base_albert'
tokenizer = BertTokenizer.from_pretrained(MODEl_NAME, do_lower_case=True)
stand_data_path = './data/standard_data.csv'

def str_q2b(ustring):
    # 全角字符转半角字符
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 0x3000:
            inside_code = 0x0020
        elif 0xFF01 <= inside_code <= 0xFF5E:
            inside_code -= 0xfee0
        else:
            inside_code = inside_code
        rstring += chr(inside_code)
    return rstring

def load_label_stand_dict(stand_data_path):
    label_stand_dict = collections.defaultdict(list)
    with open(stand_data_path) as f:
        for line in f:
            label1, label2, content = line.strip().split(',')
            content = HanziConv.toSimplified(content)
            stand = tokenizer.tokenize(str_q2b(content))
            label_stand_dict[label2].append(' '.join(stand))
    return label_stand_dict

if __name__ == "__main__":
    label_stand_dict = load_label_stand_dict(stand_data_path)
    for line in sys.stdin:
        token = line.strip().split('\t')
        label = token[2]
        if label in label_stand_dict and token[1] not in label_stand_dict[label]:
            token[1] = label_stand_dict[label][0]
        print('\t'.join(token))
