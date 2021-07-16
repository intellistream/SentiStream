import pandas as pd #著名数据处理包
import nltk 
from nltk import word_tokenize #分词函数
from nltk.corpus import stopwords #停止词表，如a,the等不重要的词
from nltk.corpus import sentiwordnet as swn #得到单词情感得分
import string #本文用它导入标点符号，如!"#$%& 
def lexicon_sentiment(text):
    stop = stopwords.words("english") + list(string.punctuation)
    ttt = nltk.pos_tag([i for i in word_tokenize(str(text).lower()) if i not in stop])
    n = ['NN','NNP','NNPS','NNS','UH']
    verb = ['VB','VBD','VBG','VBN','VBP','VBZ']
    a = ['JJ','JJR','JJS']
    r = ['RB','RBR','RBS','RP','WRB']
    word_form = []
    for key in ttt:
        if key[1] in n:
            word_form.append(f'{key[0]}.n.01')
        elif key[1] in verb:
            word_form.append(f'{key[0]}.v.01')
        elif key[1] in a:
            word_form.append(f'{key[0]}.a.01')
        elif key[1] in r:
            word_form.append(f'{key[0]}.r.01')
        else:
            word_form.append('')
    pos_score,neg_score = 0,0
    for word in word_form:
        try:
            pos_score += swn.senti_synset(word).pos_score()
            neg_score += swn.senti_synset(word).neg_score()
        except:
            pos_score += 0
    if pos_score >= neg_score:
        print('positive',pos_score,neg_score)
    else:
        print('negative',pos_score,neg_score)
lexicon_sentiment('wonderful moive')
