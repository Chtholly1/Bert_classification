# -*-coding:utf-8-*-
import os
import re
import random
import logging
import argparse
import warnings
#import datetime
import collections
from datetime import date,timedelta,datetime
import torch as t
import torch.nn as nn
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader

from business.model_plant import train, validate, test
from business.dataprocess.data_utils import MyDataSet, load_data
from business.models.model import AlBertModelSCL, AlBertModel
from business.tools import EMA, FGM, setup_seed
from conf.config import Config, args, MODEl_NAME

logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s')
warnings.filterwarnings(action='ignore')

def test1(args):
    setup_seed(2000)
    device = t.device("cuda:{}".format(args.gpu_index) if t.cuda.is_available() else "cpu")
    tokenizer = BertTokenizerFast.from_pretrained(MODEl_NAME, do_lower_case=True)

    logging.info("\t* Loading test data...")
    train_data = load_data(args.train_file)
    train_dataset = MyDataSet(train_data, tokenizer)
    train_loader = DataLoader(train_dataset, shuffle=False, batch_size=args.batch_size)

    logging.info("\t* Building model...")
    config = Config()
    #model = BertModel(config).to(device)
    if args.use_SCL_loss:
        model = AlBertModelSCL(config, device).to(device)
    else:
        model = AlBertModel(config).to(device)
    checkpoint = t.load(args.load_model, map_location=device)['model']
    state_dict_T = checkpoint
    model.load_state_dict(state_dict_T, strict=True)
    model.eval()

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    #yesterday = date.today() + timedelta(days=-1)
    date_time =  datetime.now().strftime("%Y%m%d%H")
    output_file = os.path.join(args.result_dir, 'test.' + date_time + '.txt')
    epoch_time, epoch_loss, epoch_accuracy = test(model, train_loader, train_data, device, output_file=output_file)
    logging.info("-> Valid. time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%\n"
                 .format(epoch_time, epoch_loss, (epoch_accuracy * 100)))


if __name__ == '__main__':

    test1(args)
