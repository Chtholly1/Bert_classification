# -*-coding:utf-8-*-
import os
import re
import random
import logging
import argparse
import warnings
import collections
from datetime import date,timedelta,datetime
import torch as t
import torch.nn as nn
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader

from business.model_plant import train, validate, test
from business.dataprocess.data_utils import MyDataSet, generate_all_stand, generate_all_data, generate_data
from business.models.model import AlBertModelCNN, BertModelCNN, BertModel, AlBertModel, AlbertCNN, AlBertModelNewLoss
from business.tools import EMA, FGM, setup_seed
from conf.config import Config, args, MODEl_NAME

from conf.label_black_word import bw_dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s')
warnings.filterwarnings(action='ignore')

def test1(args):
    stand_dict = generate_all_stand(args.train_file)
    data_list = generate_all_data(args.input_file, stand_dict)
    
    device = t.device("cuda:{}".format(args.gpu_index) if t.cuda.is_available() else "cpu")
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name, do_lower_case=True)

    logging.info(20 * "=" + " Loading model from {} ".format(args.model) + 20 * "=")
    logging.info("\t* Loading validation data...")
    
    val_dataset = MyDataSet(data_list, args.max_length, tokenizer)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size)

    logging.info("\t* Building model...")
    config = Config()
    model = AlBertModel(config).to(device)
    checkpoint = t.load(args.model, map_location=device)['model']
    #matrix = checkpoint.values()
    #name = [i[7:] for i in checkpoint.keys()]
    #state_dict_T = dict(zip(name,matrix))
    state_dict_T = checkpoint
    model.load_state_dict(state_dict_T, strict=True)
    model.eval()

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    date_time =  datetime.now().strftime("%Y%m%d%H")
    output_file = os.path.join(args.result_dir, 'test.' + date_time + '.csv')
    epoch_time, epoch_loss, epoch_accuracy, epoch_auc = test(model, val_loader, data_list, device, output_file=output_file)
    logging.info("-> Valid. time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%, auc: {:.4f}\n"
                 .format(epoch_time, epoch_loss, (epoch_accuracy * 100), epoch_auc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_file', type=str, default='./resource/data/talk_to_stand.train_16.28k')
    parser.add_argument('--input_file', type=str, default='./resource/data/talk_to_stand.val_16.7k')
    parser.add_argument('--model', type=str, default='./resource/models/albert_model.92.1221')
    parser.add_argument('--model_name', type=str, default=MODEl_NAME)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--gpu_index', type=int, default=1)
    parser.add_argument('--result_dir', type=str, default='./resource/result')

    test1(parser.parse_args())
