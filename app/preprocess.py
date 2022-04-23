from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from typing import OrderedDict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import pandas as pd
import re

# StopwordList = open('./Stopword-List.txt', 'r')

# stopwords = ""

# for each in StopwordList:
#     stopwords = stopwords + each
# stopwords = word_tokenize(stopwords)

# print(stopwords)

# for i in range(1, 2):
#     doc = open('./Abstracts/' + str(i) + '.txt', 'r')
#     tokens = ""
#     for line in doc:
#         tokens = tokens + line
#     tokens = word_tokenize(tokens)

#     print(tokens)
#     print()

#     lemmatizer = WordNetLemmatizer()
#     lemma = [lemmatizer.lemmatize(each).casefold() for each in tokens]

#     lemma_without_sw = [each for each in lemma if not each in stopwords]

#     for each in lemma_without_sw:
#         ''.join(e for e in each if e.isalnum())

#     punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

#     for each in lemma_without_sw:
#         if each in punctuations:
#             lemma_without_sw.remove(each)
#         each = re.sub(r'[^\w\s]','', each)
    
#     print(lemma_without_sw)
#     print(i)



class VectorSpaceModel:
    def __init__(self):
        self.numOfDocs = 448
        # self.stopwords = ""
        # self.tokens = ""
        self.lemmatizer = nltk.WordNetLemmatizer()

    def getStopWordsFromFile(self):
        fp = open('./Stopword-List.txt', 'r')
        stopwords = fp.read()
        stopwords = nltk.word_tokenize(stopwords)
        fp.close()
        return (stopwords)

    def getTokensFromFiles(self, filename):
        punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        fp = open(filename, 'r')
        text = fp.read().lower()
        text = re.sub(punctuations, "", text)
        tokens = nltk.word_tokenize(text)
        fp.close()
        return (tokens)
    
    def 