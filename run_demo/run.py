# coding:utf-8
# @Email: wangguisen@infinities.com.cn
# @Time: 2022/9/8 10:46 上午
# @File: run.py
'''
run model
'''
import torch
from torch.autograd import Variable
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchkeras import summary
import os
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
from importlib import import_module
import argparse

from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cuda')
# gpus = [0, 1]

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True  # 保证每次结果一样

class TotalConfig():
    '''
    total config
    :param train_dir: 训练集
    :param val_dir: 验证集
    :param vocab_dir: 词表
    :param sougou_pretrain_dir: 搜狗新闻预训练emb
    :param filename_trimmed_dir: 提取搜狗预训练词向量
    :param device: 单卡/CPU/单击多卡
    :param gpus: 单击多卡
    :param emb_way: emb初始化方式，搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    :param word_tokenizer: 以词为单位构建词表(数据集中词之间以空格隔开)
    :param char_tokenizer: 以字为单位构建词表
    :param emb_dim:
    :param pad_size: 每句话处理成的长度(短填长切)
    :param num_classes: 类别数
    :param n_vocab: 词表大小，运行时赋值
    :param epochs:
    :param batch_size:
    :param learning_rate:
    '''
    def __init__(self, model_name='TextCNN', doka_device=False, gpus=None):
        '''
        :param doka_device: True：一机多卡，False：单卡/cpu
        :param gpus: 多卡
        '''
        self.train_dir = "./data/train_demo.txt"
        self.val_dir = "./data/val_demo.txt"
        self.vocab_dir = "./data/vocab_demo.pkl"
        self.sougou_pretrain_dir = "./data/sgns.sogou.char"
        self.filename_trimmed_dir = "./data/embedding_SougouNews_demo"
        self.target_names = [x.strip() for x in open('./data/class_demo.txt', encoding='utf-8').readlines()]

        self.model_name = model_name
        self.save_path = './saved_dict/{}.ckpt'
        self.log_path = './log/{}'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
            if doka_device is False else torch.device('cuda')
        self.gpus = gpus

        self.emb_way = 'embedding_SougouNews.npz'
        # self.emb_way = 'random'

        self.word_tokenizer = lambda x: x.split(' ')
        self.char_tokenizer = lambda x: [y for y in x]

        self.emb_dim = 300
        self.pad_size = 32   # seq_len
        self.num_classes = 5
        self.n_vocab = 0
        self.epochs = 3
        self.batch_size = 2
        self.learning_rate = 1e-3
        self.metric_name = ['accuracy']

def get_dataloader(train_dir, val_dir, vocab_dir, pad_size, tokenizer, batch_size):
    start_time = time.time()
    print("Loading DataLoader...")

    # 读词表
    vocab = pkl.load(open(vocab_dir, 'rb'))

    train_set = MyDataSet(path=train_dir, vocab_file=vocab, pad_size=pad_size, tokenizer=tokenizer)
    val_set = MyDataSet(path=val_dir, vocab_file=vocab, pad_size=pad_size, tokenizer=tokenizer)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    time_dif = get_time_dif(start_time)
    print("Loading DataLoader Time usage:", time_dif)

    return train_loader, val_loader

if __name__ == '__main__':
    print()

    tcfig = TotalConfig()

    '''构建词表'''
    vocab, n_vocab = get_vocab(tcfig.train_dir, tokenizer=tcfig.char_tokenizer, vocab_dir=tcfig.vocab_dir)
    tcfig.n_vocab = n_vocab
    # '''提取预训练词向量'''
    # get_pre_emb(vocab, tcfig.emb_dim, tcfig.sougou_pretrain_dir, tcfig.filename_trimmed_dir)

    '''DataLoader'''
    train_loader, val_loader = get_dataloader(tcfig.train_dir, tcfig.val_dir, tcfig.vocab_dir, tcfig.pad_size, tcfig.char_tokenizer, tcfig.batch_size)

    '''init model'''
    model_name = 'TextRNN'
    tcfig.model_name = model_name
    tcfig.save_path = tcfig.save_path.format(model_name)
    tcfig.log_path = tcfig.log_path.format(model_name)
    obj = import_module('models.' + model_name)
    config = obj.Config(emb_way=tcfig.emb_way, emb_dim=tcfig.emb_dim, num_classes=tcfig.num_classes, n_vocab=tcfig.n_vocab)
    net = obj.TextRNN(config=config)

    print(net)

    '''train'''
    dfhistory = train_model(net=net, train_loader=train_loader, val_loader=val_loader, config=tcfig, model_name=model_name)

    '''eval'''
    evaluate_model(net=net, text_loader=val_loader, config=tcfig, show=True)

    '''test'''
    torch.save(net.state_dict(), tcfig.save_path)

    net.load_state_dict(torch.load(tcfig.save_path))
    content = '细节显品质 热门全能型实用本本推荐'
    pre_probs, pre = predict(net=net, content=content, vocab=vocab, tcfig=tcfig)
    print(pre_probs, pre)









