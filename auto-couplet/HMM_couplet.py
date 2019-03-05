# -*- coding: utf-8 -*-
import jieba
import jieba.posseg
from nltk import bigrams, FreqDist, ConditionalFreqDist
from collections import defaultdict
import random
import pypinyin
import os
import pickle
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from scipy.misc import imread
from harvesttext import HarvestText
import networkx as nx
from networkx.readwrite import json_graph
from pyecharts import Graph


# 判断平仄
def tone(c):  # 平为True  仄为False
    res = pypinyin.pinyin(c, style=pypinyin.TONE2)
    res = res[0][0]
    for i in res:
        if i.isdigit():
            if i == '1' or i == '2':
                return True
            else:
                return False
    return False


class Process:
    def __init__(self):
        self.pos_prob = defaultdict(float)  # 词性转移概率，bigram -> prob
        self.word_prob = defaultdict(float)  # 产生字词概率，(pos, word) -> prob
        self.refer_prob = defaultdict(float)  # 上下联对应的概率， (word, word) -> prob
        self.trans_prob = defaultdict(float)  # 词与词之间转移的概率
        self.shanglian = []
        self.cfd_refer = ConditionalFreqDist()
        self.cfd_single_refer = ConditionalFreqDist()
        self.cfd_trans_word = ConditionalFreqDist()
        self.load_data()
        self.all_word = []  # 用于统计词频，生成词云图

    def train_model(self):
        self.get_state_trans()
        with open('pos_prob.pkl', 'wb') as of1:
            pickle.dump(self.pos_prob, of1)
        with open('word_prob.pkl', 'wb') as of2:
            pickle.dump(self.word_prob, of2)
        with open('refer_prob.pkl', 'wb') as of3:
            pickle.dump(self.refer_prob, of3)
        with open('trans_prob.pkl', 'wb') as of4:
            pickle.dump(self.trans_prob, of4)
        with open('cfd_refer.pkl', 'wb') as of5:
            pickle.dump(self.cfd_refer, of5)
        with open('cfd_trans_word.pkl', 'wb') as of6:
            pickle.dump(self.cfd_trans_word, of6)
        with open('cfd_single_refer.pkl', 'wb') as of7:
            pickle.dump(self.cfd_single_refer, of7)

    def load_model(self, filename):
        with open(filename, 'rb') as if1:
            return pickle.load(if1)

    def load_data(self):
        if not os.path.exists('pos_prob.pkl'):
            self.train_model()
        else:
            self.pos_prob = self.load_model('pos_prob.pkl')
            self.word_prob = self.load_model('word_prob.pkl')
            self.refer_prob = self.load_model('refer_prob.pkl')
            self.trans_prob = self.load_model('trans_prob.pkl')
            self.cfd_refer = self.load_model('cfd_refer.pkl')
            self.cfd_trans_word = self.load_model('cfd_trans_word.pkl')
            self.cfd_single_refer = self.load_model('cfd_single_refer.pkl')

    def get_data(self):
        with open('first.txt', 'r', encoding='utf-8') as f:
            shanglian = f.readlines()
            for i in range(len(shanglian)):
                shanglian[i] = shanglian[i].rstrip('\n')

        with open('next.txt', 'r', encoding='utf-8') as f:
            xialian = f.readlines()
            for i in range(len(xialian)):
                xialian[i] = xialian[i].rstrip('\n')
        return shanglian, xialian

    def cut_and_pos(self,sent):
        # seg_list = jieba.cut(sent, cut_all=False)
        # ls = ','.join(seg_list)
        # print(ls)
        # print(lst)
        sent_pos = jieba.posseg.cut(sent)
        reverse_pos = [(w.flag, w.word) for w in sent_pos]
        pure_pos = [word[0] for word in reverse_pos]
        pure_word = [word[1] for word in reverse_pos]
        pure_pos.insert(0, 'START')
        pure_pos.append('END')
        # print(sent_pos, pure_pos)
        return sent_pos, pure_pos, pure_word, reverse_pos

    def get_state_trans(self):
        shanglian, xialian = self.get_data()
        all_pos = []
        pos_bigram = []
        trans_bigram = []
        single_refer = []
        pos_word = []
        refer_bigram = []
        for i in range(len(shanglian)):
            sent1 = shanglian[i]
            sent2 = xialian[i]
            sent1_pos, sent1_pure_pos, pure_word1, reverse_pos1 = self.cut_and_pos(sent1)
            sent2_pos, sent2_pure_pos, pure_word2, reverse_pos2 = self.cut_and_pos(sent2)
            all_pos = all_pos + sent1_pure_pos + sent2_pure_pos
            self.all_word = self.all_word + pure_word1 + pure_word2
            trans_bigram = trans_bigram + list(bigrams(pure_word1)) + list(bigrams(pure_word2))

            if len(pure_word1) == len(pure_word2):  # 上下联对应
                refer_bigram = refer_bigram + [(pure_word1[i], pure_word2[i]) for i in range(len(pure_word1)) if
                                               len(pure_word1[i]) == len(pure_word2[i])]
            # 防止上下联没有形成很好对应，直接切成单字也进行对应
            l1 = list(sent1)
            l2 = list(sent2)
            # trans_bigram = trans_bigram + list(bigrams(l1)) + list(bigrams(l2))  # 把每联的单字也给组合起来形成bigram
            single_refer = single_refer + [(l1[i], l2[i]) for i in range(len(sent1))]
            pos_bigram = pos_bigram + list(bigrams(sent1_pure_pos)) + list(bigrams(sent2_pure_pos))
            pos_word = pos_word + reverse_pos1 + reverse_pos2

        freq_pos = FreqDist(all_pos)
        freq_pos_bigram = FreqDist(pos_bigram)
        cfd_word_pos = ConditionalFreqDist(pos_word)
        self.cfd_refer = ConditionalFreqDist(refer_bigram)
        self.cfd_single_refer = ConditionalFreqDist(single_refer)
        self.cfd_trans_word = ConditionalFreqDist(trans_bigram)
        for i in freq_pos_bigram.keys():
            self.pos_prob[i] = freq_pos_bigram[i] / freq_pos[i[0]]
        for i in freq_pos.keys():
            if i == 'START' or i == 'END':
                continue
            for j in cfd_word_pos[i].keys():
                self.word_prob[(i,j)] = cfd_word_pos[i][j]/freq_pos[i]
        for i in self.cfd_refer.keys():
            count = sum(self.cfd_refer[i][k] for k in self.cfd_refer[i].keys())
            for j in self.cfd_refer[i].keys():
                self.refer_prob[(i,j)] = self.cfd_refer[i][j]/count
        for i in self.cfd_trans_word.keys():
            count = sum(self.cfd_trans_word[i][k] for k in self.cfd_trans_word[i].keys())
            for j in self.cfd_trans_word[i].keys():
                self.trans_prob[(i,j)] = self.cfd_trans_word[i][j]/count

    def gen_xialian(self, raw):  # 传入上联分词结果
        dic = defaultdict(list)
        shanglian = []
        for (i, word) in enumerate(raw):
            if word not in self.cfd_refer.keys():  # 没有在训练数据中找到这个词
                chars = list(word)
                for char in chars:
                    for j in self.cfd_single_refer[char].keys():
                        dic[char].append(j)
                    shanglian.append(char)
            else:
                shanglian.append(word)
                for(base, refer) in self.refer_prob.keys():
                    if base == word:
                        dic[base].append(refer)
        """
        # 根据已有的上联形成下联，直接取最为对仗的词即可
        print(dic)
        res = ''
        for word in shanglian:
            max_prob = 0
            max_res = dic[word][0]
            for refer in dic[word]:
                if self.refer_prob[(word, refer)] > max_prob:
                    max_prob = self.refer_prob[(word, refer)]
                    max_res = refer
            res = res + max_res
        return res
        """

        temp = []
        temp_prob = []
        for i in range(10):  # 形成十个下联，选择概率较高的前五个
            res = []
            for word in shanglian:
                index = min(int(random.random() * len(dic[word])), len(dic[word])-1)
                res.append(dic[word][index])
            temp.append(res)
        prob_dict = defaultdict(float)
        for i in range(len(temp)):
            sent = temp[i]
            last_word = sent[-1][-1]
            if not tone(last_word):  # 下联要求是平声
                continue
            sent_prob = max(self.refer_prob[(shanglian[0], sent[0])], 0.000001)  # 对拆分的单字加以惩罚
            for j in range(1, len(shanglian)):
                sent_prob = sent_prob * max(0.000001, self.trans_prob[(sent[j-1], sent[j])]) * \
                            max(self.refer_prob[(shanglian[j], sent[j])], 0.000001)
            prob_dict[''.join(sent)] = sent_prob
        prob_dict = sorted(prob_dict.items(), key=lambda item:item[1], reverse=True)
        for item in prob_dict:
            print(item[0], ' ', item[1])
        return prob_dict[0][0]

    def gen_shanglian(self, start, p, length):  # 指定上联长度为length，用于生成对联
        if start not in self.cfd_trans_word.keys():
            return False
        nxt = list(self.cfd_trans_word[start].keys())
        random.shuffle(nxt)
        for choice in nxt:
            if len(choice) + p > length:
                return False
            if len(choice) + p == length:  # 刚好生成成功
                self.shanglian.append(choice)
                return True
            self.shanglian.append(choice)
            q = len(choice) + p
            if self.gen_shanglian(choice, q, length):
                return True
            self.shanglian.pop()
        return False

    def Gen_shanglian(self, start, length):  # 封装了递归函数
        prob_dict = defaultdict(float)
        for i in range(10):  # 生成十句比较好的句子
            if self.gen_shanglian(start, len(start), length):
                sent = self.shanglian[:]
                sent.insert(0, start)
                last_word = sent[-1][-1]  # 获取最后一个字，判断平仄
                if not tone(last_word):  # 若是平声，不合要求
                    sent_prob = 1
                    for j in range(1, len(sent)):
                        pos1 = list(jieba.posseg.cut(sent[j - 1]))[0].flag
                        pos2 = list(jieba.posseg.cut(sent[j]))[0].flag
                        sent_prob = sent_prob * max(0.000001, self.trans_prob[(sent[j - 1], sent[j])]) * \
                                    max(0.000001, self.pos_prob[(pos1, pos2)])
                    prob_dict[''.join(sent)] = sent_prob
                self.shanglian.clear()

        prob_dict = sorted(prob_dict.items(), key=lambda item: item[1], reverse=True)
        for item in prob_dict:
            print(item[0], ' ', item[1])
        self.shanglian = prob_dict[0][0]
        return self.shanglian

    def Dui_duilian(self, shanglian):
        raw = jieba.posseg.cut(shanglian)
        raw = [item.word for item in raw]
        xialian = self.gen_xialian(raw)
        print(shanglian, ',', xialian)

    def Gen_duilian(self, start, length):
        shanglian = self.Gen_shanglian(start, length)
        self.Dui_duilian(shanglian)


test = Process()
test.Dui_duilian('春风轻拂千山绿')
test.Gen_duilian('江山', 7)




