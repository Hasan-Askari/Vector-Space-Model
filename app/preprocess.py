from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from typing import OrderedDict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

StopwordList = open('./Stopword-List.txt', 'r')

stopwords = ""

for each in StopwordList:
    stopwords = stopwords + each
stopwords = word_tokenize(stopwords)

print(stopwords)

for i in range(1, 2):
    doc = open('./Abstracts/' + str(i) + '.txt', 'r')
    tokens = ""
    for line in doc:
        tokens = tokens + line
    tokens = word_tokenize(tokens)

    print(tokens)
    print()

    lemmatizer = WordNetLemmatizer()
    lemma = [lemmatizer.lemmatize(each).casefold() for each in tokens]

    lemma_without_sw = [each for each in lemma if not each in stopwords]

    for each in lemma_without_sw:
        ''.join(e for e in each if e.isalnum())

    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

    for each in lemma_without_sw:
        if each in punctuations:
            lemma_without_sw.remove(each)
        each = re.sub(r'[^\w\s]','', each)
    
    print(lemma_without_sw)
    print(i)
