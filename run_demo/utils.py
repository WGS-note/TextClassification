# coding:utf-8
# @Email: wangguisen@infinities.com.cn
# @Time: 2022/9/8 10:24 上午
# @File: utils.py
'''
封装文本分类工具类
+ 构建词表
+ 提取预训练词向量
+ MyDataSet
+ train and eval
'''
import torch
from torch.autograd import Variable
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np, pandas as pd
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
from tensorboardX import SummaryWriter
from sklearn import metrics
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score

'''=========================================================================================='''

MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding

'''=========================================================================================='''

class MyDataSet(Dataset):
    def __init__(self, path, vocab_file, pad_size, tokenizer):
        self.contents = []
        self.labels = []
        # 读取数据
        file = open(path, 'r')
        for line in file.readlines():
            line = line.strip().split('\t')
            content, label = line[0], line[1]
            self.contents.append(content)
            self.labels.append(label)
        file.close()
        self.pad_size = pad_size
        self.tokenizer = tokenizer
        self.vocab = vocab_file

    def __len__(self):
        return len(self.contents)

    # 该函数是返回单条样本
    def __getitem__(self, idx):
        content, label = self.contents[idx], self.labels[idx]
        token = self.tokenizer(content)
        seq_len = len(token)
        words_line = []

        # 数据预处理的时候统一padding
        # 如果当前句子小于指定的长度，就补长
        if len(token) < self.pad_size:
            token.extend([PAD] * (self.pad_size - len(token)))
        else:
            # 如果不是的话，就截断
            token = token[:self.pad_size]
            seq_len = self.pad_size

        for word in token:
            words_line.append(self.vocab.get(word, self.vocab.get(UNK)))

        tensor = torch.Tensor(words_line).long()
        label = int(label)
        seq_len = int(seq_len)
        return (tensor, seq_len), label

'''=========================================================================================='''

'''   获取已使用时间   '''
def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

'''   构建词表   '''
def build_vocab(file_path, tokenizer, max_size, min_freq):
    '''
    构建词表
    exp：{' ': 0, '0': 1, '1': 2, '2': 3, '：': 4, '大': 5, '国': 6, '图': 7, '(': 8, ')': 9, '3': 10, '人': 11, '年': 12, '5': 13, '中': 14, '新': 15,...
    :param file_path: 数据路径
    :param tokenizer: 构建词表的切分方式：word level（以词为单位构建，词之间以空格隔开）、char level（以字为单位构建）
    :param max_size: 词表长度限制
    :param min_freq: 词频阈值，小于阈值的不放入词频，即低频过滤了
    :return:
    '''
    vocab_dic = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        # 读取每行
        for line in tqdm(f):
            lin = line.strip()  # 去掉最后的\n
            if not lin:  # 如果是空格直接跳过
                continue
            content = lin.split('\t')[0]  # 文本数据用\t分割，第一个[0]为文本，[1]为标签
            for word in tokenizer(content):
                # 频次字典，如果有就返回结果出现次数+1，没有就是1
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        # 降序，截取到max_size
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
        # 生成词表字典
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        # 更新两个字符：unk、pad
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic

'''   获取&保存词表   '''
def get_vocab(train_dir, tokenizer, vocab_dir):
    if os.path.exists(vocab_dir):
        word_to_id = pkl.load(open(vocab_dir, 'rb'))
    else:
        word_to_id = build_vocab(train_dir, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(word_to_id, open(vocab_dir, 'wb'))

    return word_to_id, len(word_to_id)

'''   提取预训练词向量   '''
def get_pre_emb(vocab, emb_dim, pretrain_dir, filename_trimmed_dir):
    if os.path.exists(filename_trimmed_dir):
        return None
    embeddings = np.random.rand(len(vocab), emb_dim)
    f = open(pretrain_dir, "r", encoding='UTF-8')
    for i, line in enumerate(f.readlines()):
        # if i == 0:  # 若第一行是标题，则跳过
        #     continue
        lin = line.strip().split(" ")
        if lin[0] in vocab:
            idx = vocab[lin[0]]
            # emb = [float(x) for x in lin[1:301]]
            emb = [float(x) for x in lin[1:emb_dim + 1]]
            embeddings[idx] = np.asarray(emb, dtype='float32')
    f.close()
    # 以.npz压缩保存
    np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)

    print(embeddings.shape)

'''   权重初始化，默认xavier   '''
def init_network(model, method='xavier', exclude='embedding', seed=123):
    '''
    权重初始化，默认xavier
    :param model: net
    :param method: 初始化方法：xavier、kaiming、normal_
    :param exclude:
    :param seed:
    :return:
    '''
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass

'''=========================================================================================='''

def train_model(net, train_loader, val_loader, config, model_name):
    '''
    :param net:
    :param train_loader:
    :param val_loader:
    :param config:
    :return:
    '''
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1, last_epoch=-1)
    early_stopping = EarlyStopping(savepath='./data/{}_checkpoint.pt'.format(model_name), patience=1)

    start_time = time.time()
    print("\n" + "********** start training **********")

    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    columns = ["epoch", "loss", * config.metric_name, "val_loss"] + ['val_' + mn for mn in config.metric_name]
    dfhistory = pd.DataFrame(columns=columns)

    for epoch in range(1, config.epochs + 1):
        '''   训练   '''
        print("Epoch {0} / {1}".format(epoch, config.epochs))
        step_start = time.time()
        step_num = 0
        train_loss, train_probs, train_y, train_pre = [], [], [], []
        net.train()
        for batch, (x, y) in enumerate(train_loader):
            step_num += 1
            optimizer.zero_grad()
            pred_probs = net(x)

            loss = loss_function(pred_probs, y)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            train_probs.extend(pred_probs.tolist())
            train_y.extend(y.tolist())
            # train_pre.extend(torch.where(pred_probs > 0.5, torch.ones_like(pred_probs), torch.zeros_like(pred_probs)))
            train_pre.extend(torch.argmax(pred_probs.data, dim=1))

            if step_num % 100 == 0:
                step_train_acc = accuracy_score(y_true=y.tolist(), y_pred=torch.argmax(pred_probs.data))
                writer.add_scalar('loss/train', loss.item(), step_num)
                writer.add_scalar('acc/val', step_train_acc, step_num)

        '''   验证   '''
        val_loss, val_probs, val_y, val_pre = [], [], [], []
        net.eval()
        with torch.no_grad():
            for batch, (x, y) in enumerate(val_loader):
                pred_probs = net(x)
                loss = loss_function(pred_probs, y)

                val_loss.append(loss.item())
                val_probs.extend(pred_probs.tolist())
                val_y.extend(y.tolist())
                # val_pre.extend(torch.where(pred_probs > 0.5, torch.ones_like(pred_probs), torch.zeros_like(pred_probs)))
                val_pre.extend(torch.argmax(pred_probs.data, dim=1))

        '''  一次epoch结束 记录日志   '''
        epoch_loss, epoch_val_loss = np.mean(train_loss), np.mean(val_loss)
        # train_auc = roc_auc_score(y_true=train_y, y_score=train_probs)
        train_acc = accuracy_score(y_true=train_y, y_pred=train_pre)
        # val_auc = roc_auc_score(y_true=val_y, y_score=val_probs)
        val_acc = accuracy_score(y_true=val_y, y_pred=val_pre)

        # dfhistory.loc[epoch - 1] = (epoch, epoch_loss, train_acc, train_auc, epoch_val_loss, val_acc, val_auc)
        dfhistory.loc[epoch - 1] = (epoch, epoch_loss, train_acc, epoch_val_loss, val_acc)

        step_end = time.time()
        # print("step_num: %s - %.1fs - loss: %.5f   accuracy: %.5f   auc: %.5f - val_loss: %.5f   val_accuracy: %.5f   val_auc: %.5f"% (step_num, (step_end - step_start) % 60, epoch_loss, train_acc, train_auc, epoch_val_loss, val_acc, val_auc))
        print("step_num: %s - %.1fs - loss: %.5f   accuracy: %.5f   - val_loss: %.5f   val_accuracy: %.5f"% (step_num, (step_end - step_start) % 60, epoch_loss, train_acc, epoch_val_loss, val_acc))

        # if scheduler is not None:
        #     scheduler.step()
        #
        # if early_stopping is not None:
        #     early_stopping(epoch_val_loss, net)
        #     if early_stopping.early_stop:
        #         print("Early stopping")
        #         break

    writer.close()
    end_time = time.time()
    print('********** end of training run time: {:.0f}分 {:.0f}秒 **********'.format((end_time - start_time) // 60, (end_time - start_time) % 60))
    print()
    return dfhistory

def evaluate_model(net, text_loader, config, device='cpu', show=False):
    '''
    测试集评估demo
    待修改：注释部分多分类评估
    :param net:
    :param text_loader:
    :param config:
    :param device:
    :param show:
    :return:
    '''
    val_loss, val_probs, val_y, val_pre = [], [], [], []
    net.eval()
    with torch.no_grad():
        for batch, (x, y) in enumerate(text_loader):
            pred_probs = net(x)
            loss = F.cross_entropy(pred_probs, y)
            val_loss.append(loss.item())
            val_probs.extend(pred_probs.tolist())
            val_y.extend(y.tolist())
            val_pre.extend(torch.argmax(pred_probs.data, dim=1))

    acc_condition, precision_condition, recall_condition = accDealWith2(val_y, val_pre)
    # precision = np.around(metrics.precision_score(val_y, val_pre), 4)
    # recall = np.around(metrics.recall_score(val_y, val_pre), 4)
    # accuracy = np.around(metrics.accuracy_score(val_y, val_pre), 4)
    # f1 = np.around(metrics.f1_score(val_y, val_pre), 4)

    if show:
        print('=' * 30)
        print('test eval')
        print(' loss: ', loss)
        # print(' accuracy: ', accuracy)
        # print(' precision: ', precision)
        # print(' recall: ', recall)
        # print(' f1: ', f1)
        print(' ', acc_condition)
        print(' ', precision_condition)
        print(' ', recall_condition)
        report = metrics.classification_report(val_y, val_pre, target_names=config.target_names)
        print(report)

    # return precision, recall, accuracy, f1, loss, acc_condition, precision_condition, recall_condition
    return loss, acc_condition, precision_condition, recall_condition

def predict(net, content, vocab, tcfig):
    '''
    单条预测demo
    :param net:
    :param content:
    :param vocab:
    :param tcfig:
    :return:
    '''
    token = tcfig.char_tokenizer(content)
    print(token)
    seq_len = len(token)
    words_line = []

    if len(token) < tcfig.pad_size:
        token.extend([PAD] * (tcfig.pad_size - len(token)))
    else:
        token = token[:tcfig.pad_size]
        seq_len = tcfig.pad_size
    for word in token:
        words_line.append(vocab.get(word, vocab.get(UNK)))

    x = (torch.Tensor([words_line]).long(), seq_len)
    print(x)
    out = net(x)
    outs = torch.max(out.data, dim=1)
    pre_probs, pre = outs[0], outs[1]
    return pre_probs, pre

# reference: https://github.com/Bjarten/early-stopping-pytorch
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, savepath=None, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.savepath = savepath

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        if self.savepath is not None:
            torch.save(model.state_dict(), self.savepath)
        else:
            torch.save(model.state_dict(), 'checkpoint.pt')	# 这里会存储迄今最优模型的参数

        self.val_loss_min = val_loss

def accDealWith2(y_test, y_pre):
    lenall = len(y_test)
    if type(y_test) != list:
        y_test = y_test.flatten()
    pos = 0
    pre = 0
    rec = 0
    precisoinlen = 0
    recallLen = 0

    for i in range(lenall):
        # 准确率
        if y_test[i] == y_pre[i]:
            pos += 1
        # 精确率
        if y_pre[i] == 1:
            pre += 1
            if y_test[i] == 1:
                precisoinlen += 1
        # 召回率
        if y_test[i] == 1:
            rec += 1
            if y_pre[i] == 1:
                recallLen += 1

    acc_condition = '预测对的：{}，总样本：{}'.format(pos, lenall)
    if pre != 0:
        precision_condition = '预测为正的样本数：{}，其中实际为正的样本数：{}，精确率：{}'.format(pre, precisoinlen, np.around(precisoinlen / pre, 4))
    else:
        precision_condition = '预测为正的样本数：{}，其中实际为正的样本数：{}，精确率：{}'.format(pre, precisoinlen, 0.0)

    if rec != 0:
        recall_condition = '正例样本：{}，正例中预测正确的数量：{}，召回率：{}'.format(rec, recallLen, np.around(recallLen / rec, 4))
    else:
        recall_condition = '正例样本：{}，正例中预测正确的数量：{}，召回率：{}'.format(rec, recallLen, 0.0)

    return acc_condition, precision_condition, recall_condition