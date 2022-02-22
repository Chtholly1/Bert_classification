#coding=utf-8

import argparse

MODEl_NAME='./resource/base_models/base_albert'
#MODEl_NAME='./resource/base_models/base_roberta'

categories = ['car_related',
              'appearance',
              'interior',
              'config',
              'space',
              'control',
              'comfort',
              'power',
              'energy_consumption',
              'car_use',
              'budget',
              'offer',
              'discount',
              'loan',
              'insurance',
              'final_price',
              'car_price_other',
              'other']

label_2_ids = {}
for idx, item in enumerate(categories):
    label_2_ids[item] = idx
ids_2_label = {v:k for (k,v) in label_2_ids.items()}

class Config:
    def __init__(self):
        self.num_labels = 2
        self.dropout_rate = 0.2
        self.num_attention_heads = 12
        self.attention_probs_dropout_prob = 0.1
        self.classifier_dropout_prob = 0.1
        self.hidden_size = 768
        self.final_out_size = 128
        self.vocab_size = 0
        self.embedding_size = 768
        self.out_channels = 256
        self.kernel_size = [2, 3, 4]
        self.max_text_len = 256
        self.cnn_conf_list = [ (3, 1), (3, 2), (3, 4), (3, 1)] 
        self.model_name = MODEl_NAME


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default=MODEl_NAME, type=str)
parser.add_argument("--train_file", type=str, help="训练集文件", default='resource/data/train_2.txt')
parser.add_argument("--val_file", type=str, help="验证集文件", default='resource/data/dev_2.txt')
parser.add_argument("--target_dir", default="resource/models", type=str)
parser.add_argument("--result_dir", default="resource/result/", type=str)
parser.add_argument("--load_model", default="resource/models/best.pth.tar.2022021615", type=str)
parser.add_argument("--max_length", default=256, type=int, help="截断的最大长度")
parser.add_argument("--epochs", default=25, type=int, help="最多训练多少个epoch")
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--lr", default=2.5e-5, type=int)
parser.add_argument("--max_grad_norm", default=10.0, type=int)
parser.add_argument("--patience", default=3, type=int)
parser.add_argument("--gpu_index", default=1, type=int)
parser.add_argument("--attack_type", default='FGM', type=str)
parser.add_argument("--use_EMA", default=False, type=int)
parser.add_argument("--use_SCL_loss", default=True, type=int)
parser.add_argument("--use_mixup", default=False, type=int)

args = parser.parse_args()

