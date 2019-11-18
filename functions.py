# -*- coding: utf-8 -*-
import re
import nltk
from nltk.tokenize import wordpunct_tokenize

def get_freq_dist_list(tokens):
    ls = []

    for tk_line in tokens:
        for word in tk_line:
            ls.append(word)

    return ls

def get_text_cloud(tokens):
    text = ''

    for tk_line in tokens:
        for word in tk_line:
            text += word + ' '
        
    return text

def untokenize_text(tokens):
    ls = []

    for tk_line in tokens:
        new_line = ''
        
        for word in tk_line:
            new_line += word + ' '
            
        ls.append(new_line)
        
    return ls



def remove_url(data):
    data_aux = []
    regexp1 = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    regexp2 = re.compile('www?.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    
    for line in data:
        urls = regexp1.findall(line)

        for u in urls:
            line = line.replace(u, ' ')

        urls = regexp2.findall(line)

        for u in urls:
            line = line.replace(u, ' ')
            
        data_aux.append(line)
    return data_aux


def remove_char(data):
    data_aux = []
    regexp1 = re.compile("[^\w\sÀ-ÿ]")
    
    for line in data:
        urls = regexp1.findall(line)

        for u in urls:
            line = line.replace(u, "")
            
        data_aux.append(line)
    return data_aux

def remove_regex(data, regex_pattern):
    data_aux = []
    for line in data:
        matches = re.finditer(regex_pattern, line)
        for m in matches: 
            line = re.sub(m.group().strip(), '', line)

        data_aux.append(line)

    return data_aux

#def replace_emoticons(data, emoticon_list):
#    ls = []
#    for line in data:
#        line = re.sub(':\)|;\)|:-\)|\(-:|:-D|=D|:P|xD|X-p|\^\^|:-*|\^\.\^|\^\-\^|\^\_\^|\,-\)|\)-:|:\'\(|:\(|:-\(|:\S|T\.T|\.\_\.|:<|:-\S|:-<|\*\-\*|:O|=O|=\-O|O\.o|XO|O\_O|:-\@|=/|:/|X\-\(|>\.<|>=\(|D:', '', line)
#        ls.append(line)
#
#    return ls

def replace_emoticons(data):
    data_aux = []
    #emoticon_list = emoticon_list = {':))': 'positiveEmote', ':)': 'positiveEmote', ':D': 'positiveEmote', ':(': 'negativeEmote', ':((': 'negativeEmote'}
    emoticon_list = emoticon_list = {':))': '', ':)': '', ':D': '', ':(': '', ':((': ''}
    for line in data:
        for exp in emoticon_list:
            line = line.replace(exp, emoticon_list[exp])

        data_aux.append(line)

    return data_aux

def tokenize_text(data):
    data_aux = []

    for line in data:
        tokens = wordpunct_tokenize(line)
        data_aux.append(tokens)

    return data_aux

def tokenize_bigram(data):
    data_aux = []
    for line in data:
        bigramFeatureVector = []
        for item in nltk.bigrams(line.split()):
            bigramFeatureVector.append(' '.join(item))
        data_aux.append(bigramFeatureVector)
    return data_aux
    

def apply_standardization(tokens):
    ls = []
    std_list = {'eh': 'é', 'vc': 'você', 'vcs': 'vocês', 'mt' : 'muito', 'tb': 'também', 'tbm': 'também', 'obg': 'obrigado', 'gnt': 'gente', 'q': 'que', 'n': 'n~eo', 'cmg': 'comigo', 'p': 'para', 'ta': 'está', 'to': 'estou', 'vdd': 'verdade'}
    for tk_line in tokens:
        new_tokens = []
        
        for word in tk_line:
            if word.lower() in std_list:
                word = std_list[word.lower()]
                
            new_tokens.append(word) 
            
        ls.append(new_tokens)

    return ls

def get_stopwords():
    nltk_stopwords = nltk.corpus.stopwords.words('portuguese')
    slang_stopwords = ['é', 'vou', 'que', 'tão', 'ta', 'pra', 'pa', 'pá', 'vc', 'vcs', 'ta', 'tá', 'to', 'tamo', 'temo', 
                       'so', 'hey', 'gt', 'uai', 'ué', 'aí', 'ei', 'ja', 'ai', '...']
    nltk_stopwords.extend(slang_stopwords)
    nltk_stopwords = list(set(nltk_stopwords))
    return nltk_stopwords

def remove_stopwords(tokens):
    data_aux = []
    stopword_list = get_stopwords()
    for tk_line in tokens:
        new_tokens = []
        
        for word in tk_line:
            if word.lower() not in stopword_list:
                new_tokens.append(word) 
            
        data_aux.append(new_tokens)
        
    return data_aux

def apply_stemmer(tokens):
    data_aux = []
    stemmer = nltk.stem.RSLPStemmer()

    for tk_line in tokens:
        new_tokens = []
        
        for word in tk_line:
            word = str(stemmer.stem(word))
            new_tokens.append(word) 
            
        data_aux.append(new_tokens)
        
    return data_aux

def get_accuracy(matrix):
    acc = 0
    n = 0
    total = 0
    
    for i in range(0, len(matrix)):
        for j in range(0, len(matrix)):
            if(i == j): 
                n += matrix[i,j]
            
            total += matrix[i,j]
            
    acc = n / total
    return acc

