# coding:utf-8
# @Email: wangguisen@infinities.com.cn
# @Time: 2022/9/7 9:32 下午
# @File: TextRNN.py
from torch import nn
import torch
import numpy as np

class Config(object):
    '''
    :param model_name:
    :param save_path: 模型保存位置
    :param log_path: tensorboard日志查看
    :param embedding_pretrained: 预训练词向量
    :param emb_dim: 字向量维度, 若使用了预训练词向量，则维度统一
    :param dropout:
    :param num_classes:
    :param n_vocab: 词表大小
    :param hidden_size: lstm隐藏层
    :param num_layers: lstm层数
    '''
    def __init__(self, emb_way, emb_dim, num_classes, n_vocab):
        '''
        :param emb_way: emb初始化方式，搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
        :param emb_dim:
        :param num_classes:
        :param n_vocab: 词表大小
        '''

        self.embedding_pretrained = torch.tensor(np.load('./data/' + emb_way)["embeddings"].astype('float32'))\
            if emb_way != 'random' else None
        self.emb_dim = self.embedding_pretrained.size(1) \
            if self.embedding_pretrained is not None else emb_dim

        self.dropout = 0.5
        self.num_classes = num_classes
        self.n_vocab = n_vocab
        self.hidden_size = 128
        self.num_layers = 3

class TextRNN(nn.Module):
    def __init__(self, config):
        super(TextRNN, self).__init__()

        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.emb_dim, padding_idx=config.n_vocab - 1)

        # batch_first=True：batch_size在前
        self.lstm = nn.LSTM(config.emb_dim, config.hidden_size, config.num_layers, bidirectional=True, batch_first=True, dropout=config.dropout)
        self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)

    def forward(self, x):
        # x: [batch_size, seq_len]
        x, x_len = x
        out = self.embedding(x)  # [batch_size, seq_len, emb_dim]

        # out: [batch_size, seq_len, num_directions * hidden_size]
        # hn, cn: [num_directions * num_layers, batch_size, hidden_size]
        out, (hn, cn) = self.lstm(out)

        out = self.fc(out[:, -1, :])  # 句子最后时刻的 hidden state
        return out

