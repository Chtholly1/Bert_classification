#coding:utf-8
import os
import sys
from transformers import BertTokenizer, BertTokenizerFast
from hanziconv import HanziConv

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


def content_process(content, tokenizer):
    content = str_q2b(content)
    content = HanziConv.toSimplified(content)
    content = tokenizer.tokenize(content)
    return content

tokenizer = BertTokenizerFast.from_pretrained('./base_albert')
ori_talk_path = sys.argv[1]
talk_dict = dict()
#new_talk_path = sys.argv[2]
with open(ori_talk_path) as f:
    for line in f:
        stand, talk, label = line.strip().split('\t')
        new_talk = ''.join(content_process(talk, tokenizer)).replace(' ','').replace('#','').replace(',', '，').replace(';','；').replace(':', '：').replace('?', '？').replace('(', '（').replace(')', '）').lower()
        talk_dict[new_talk] = talk

for line in sys.stdin:
    token = line.strip().split()
    talk = token[0]
    if talk in talk_dict:
        token[0] = talk_dict[talk]
        print('\t'.join(token))
