# -*- coding:utf-8 -*-
import re
import os
import sys
import time
import jieba
import logging
import numpy as np
from gensim.corpora import WikiCorpus
#from opencc import OpenCC# 最终使用c++版，搁置这个python库
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

#将zhwiki.xml.bz2转化为corpus.zhwiki.txt
def parse_corpus(input_file, output_file):
    space = ' '
    i = 0
    with open(output_file, 'w', encoding='utf-8') as fout:
        wiki = WikiCorpus(input_file, lemmatize=False, dictionary={})
        for text in wiki.get_texts():
            fout.write(space.join(text) + '\n')
            i += 1
            if i % 10000 == 0:
                logger.info('{t} *** {i} \t docs has been dealed'
                .format(i=i, t=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())))

# 简繁转换
def zh_tr2simp(input_file, output_file):
    
    zh_tr_corpus = []
    with open(input_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.replace('\n', '').replace('\n', '')
            zh_tr_corpus.append(line)
    logger.info('read traditional file finished!')

    
    cc = OpenCC('t2s')
    zh_simp = []
    for i,line in zip(range(len(zh_tr_corpus)), zh_tr_corpus):
        if i%1000 == 0:
            logger.info(' *** {i} \t docs has been dealed'.format(i=i))
        zh_simp.append(cc.convert(line))
    logger.info('convert t2s finished!')

    
    with open(output_file, 'w', encoding='utf-8') as fout:
        for line in zh_simp:
            fout.writelines(line + '\n')

# 使用结巴分词分词
def fenciJ(sentence):
    
    sentence_depart = jieba.cut(sentence.strip())
    segment_res = ' '.join(sentence_depart)
    return segment_res

# 正则对字符串清洗
def zhengzeQX(str_doc):
    # 正则过滤掉特殊符号、标点、英文、数字等。
    r1 = '[0-9’!"#$%&\'()*+,-./:：;；|<=>?@，—。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    str_doc = re.sub(r1, ' ', str_doc)

    # 去掉字符
    str_doc = re.sub('\u3000', '', str_doc)

    # 去除空格
    str_doc=re.sub('\s+', ' ', str_doc)

    # 去除换行符
    #str_doc = str_doc.replace('\n',' ')
    return str_doc

# 清洗一句话的符号空格及分词
def QX(str_doc):
    str_doc = zhengzeQX(str_doc)
    str_doc = fenciJ(str_doc)
    return str_doc

# 清洗全部的符号空格及分词
def QXALL(input_file, output_file):
    #读取zhwiki文件
    zhwiki_simplified_corpus = []
    with open(input_file, 'r', encoding = 'utf-8') as fin:
        for line in fin:
            zhwiki_simplified_corpus.append(line)
    logging.info('read zhwiki finished!')

    #将zhwiki分词
    zhwiki_corpus_seg = []
    with open(output_file, 'w', encoding = 'utf-8') as fout:
        for i, line in  zip(range(len(zhwiki_simplified_corpus)), zhwiki_simplified_corpus):
            if i%1000 == 0:
                logging.info(' *** {i} \t docs has been dealed'.format(i=i))
            zhwiki_corpus_seg.append(QX(line))
    logging.info('zhwiki segment finished!')

    #写入文件zhwiki_corpus_simplified_seg.txt文件中
    with open(output_file, 'w', encoding='utf-8') as fout:
        for line in zhwiki_corpus_seg:
            fout.writelines(line + '\n')
    logging.info('write zhwiki segment finished!')

#用于对wiki_zh进行Word2vec模型的训练
def trainW(input_file):

    sentences = LineSentence(input_file)
    model = Word2Vec(sentences, size = 100, window = 2)

    #save model
    model.save("./wiki_model", binary = False)
    model.wv.save_word2vec_format("./vec", binary = False)

# 测试pku
def testW(input_file, output_file):
    word2vec_model = Word2Vec.load("./wiki_model", binary = False)
    sim_test = []
    for each in range(500):
        sim_test.append([])
    #用于生成一个词典,词典记录了有向量的字段,如果没有将结果置为OOV
    print("===正在生成字典===")
    dictionary = []
    with open("./vec", 'r') as fin_vec:
        for line in fin_vec:
            line = line.replace('.', ' ').replace(',', ' ').strip()
            dictionary.append(line.split()[0])
    with open(os.path.join(cache_dir, 'dict.json'),'w') as fout_vec:
           json.dump(dictionary, fout_vec)
    print("===字典已生成===")
    with open(os.path.join(cache_dir, 'dict.json'),'r') as f:
        dict = json.load(f)
    with open(input_file, 'r', encoding='utf-8') as fin:
        i = 0
        for line in fin:
            str_tmp1, str_tmp2 = line.split('\t', 1)
            str1 = str_tmp1
            str2 = str_tmp2.replace('\n', '')
            if (str1 not in dict) or (str2 not in dict):
                sim_test[i].append((str1, str2, 'OOV'))
                print(str1 + '\t' + str2 + '\t' +'OOV')
            else:
                sim = word2vec_model.wv.similarity(str1, str2)
                print(str1 + '\t' + str2 + '\t' + str(sim))
                sim_test[i].append((str1, str2, str(sim)))
            i += 1
    with open(output_file, 'w', encoding='utf-8') as fout:
        for each in sim_test:
            fout.writelines(str(each[0][0]) + '\t'+ str(each[0][1]) + '\t'+ str(each[0][2]) + '\n')
    

if __name__ == '__main__':
    QXALL("zhwiki_sp.txt", "zhwiki_sp_seg.txt")
    trainW("zhwiki_sp_seg.txt")
    #testW("pku_sim_test.txt", "str(2017211438).txt")