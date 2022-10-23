from genericpath import exists
import numpy as np
import json
import os
import nltk
import math
import re

class VectorSpaceModel:
    def __init__(self):
        self.numOfDocs = 448
        self.index = {}                                             # (vector space of documents)format: index = {'word1': {'TF': [a, b, c, ...], 'DF': d, 'IDF': e, 'TFxIDF': [f, g, h, ...] }'word2': {'TF': [a, b, c, ...], 'DF': d, 'IDF': e, 'TFxIDF': [f, g, h, ...] }} note: a, b, c, ... are values of tf, idf, etc
        self.documentVectors = {}                                   # (vectors of document) <doc1, doc2, doc3, ..., doc448>
        self.result = []                                            # final retrieval of documents
        self.stopwords = self.getStopWordsFromFile()                # stopwords
        self.alpha = 0.001                                          # alpha value for document filtering NOTE 0.001 was matching the Gold Query set. there was ambiguity as in assignment pdf, alpha value is given 0.05 but in GOLD Query set it was 0.001
        self.punctuations = "[.,!?:;‘’”“\"]"        
        self.docsPath = "Submission/Abstracts/{}.txt"    
        self.lemmatizer = nltk.WordNetLemmatizer()                  # nltk lemmatizer

    def getStopWordsFromFile(self):
        fp = open('Submission/files/Stopword-List.txt', 'r')                       # opening file in fp
        stopwords = fp.read()                                       # reading file in stopwords as str
        stopwords = nltk.word_tokenize(stopwords)                   # tokenizing stopwords and converting to list
        fp.close()                                                  # file closed
        return (stopwords)                 

    def getTokensFromFiles(self, __filename):
        fp = open(__filename, 'r')                                   # opening document file
        text = fp.read().lower()                                     # reading doc in text as str
        text = re.sub(self.punctuations, " ", text)                  # removing punctuations
        tokens = nltk.word_tokenize(text)                            # tokenizing text and converting to list
        tokens = [self.lemmatizer.lemmatize(each) for each in tokens]   # lemmatizing terms/tokens
        fp.close()                                                   # file closed
        return (tokens)     

    def createIndex(self):
        currentDoc = 1
        while(currentDoc <= self.numOfDocs):                         # loop to iterate all document files
            tokens = self.getTokensFromFiles(self.docsPath.format(currentDoc))  # getting tokens from each document
            for word in tokens:                                      # loop to iterate all tokens of each document
                if(word not in self.stopwords):                      # removing stopwords from tokens
                    if(word not in self.index):                      # condition to add new word in index (datatype: dict)
                        self.index[word] = {'TFs': [0]*self.numOfDocs, 'DF': 0, 'IDF': 0, 'TF-IDFs': [0]*self.numOfDocs} # making format of the dictionary for every distinct term
                        self.index[word]['TFs'][currentDoc - 1] += 1                 # counting Term Frequency of each term
                        self.index[word]['DF'] += 1                                  # counting Document Frequency of each term
                    else:
                        if(self.index[word]['TFs'][currentDoc - 1] >= 1):            # if term already in index and Term has occured previously in this document
                            self.index[word]['TFs'][currentDoc -1] += 1              # increase TF only
                        else:
                            self.index[word]['TFs'][currentDoc -1] += 1              # else if Term is occuring first time in this document
                            self.index[word]['DF'] += 1                              # increase TF and DF both
            currentDoc += 1
        for word in self.index:
            self.index[word]['IDF'] = math.log(self.index[word]['DF'], 10)/self.numOfDocs   # calculating IDF of every term using formula: IDF = log(N/DF)
        for word in self.index:
            for count in range(self.numOfDocs):
                self.index[word]['TF-IDFs'][count] = self.index[word]['TFs'][count] * self.index[word]['IDF'] # calculating TFxIDF of each term in dictionary/index
        self.saveIndex()                                              # saving to file after creating index

    def saveIndex(self):                                              # saving index to json
        print("\nSaving Index to json file...\n")
        with open('Submission/files/index.json', 'w') as file:                         # open json file as write-mood
            json.dump(self.index, file, indent=4)                     # dump/store index to json file
        print("\nIndex saved successfully!\n")
        file.close()                                                  # close file

    def loadIndex(self):                                              # loading index from file to dictionary
        print("\nLoading Index from json file...\n")    
        with open('Submission/files/index.json', 'r') as file:                         # open json file as read-only
            self.index = json.load(file)                              # load index to dictonary in python
        print("Index loading successful!\n")
        file.close()                                                  # close file

    def loadORcreateINDEX(self):
        if(len(self.index) == 0):
            if exists('Submission/files/index.json') and (os.path.getsize('Submission/files/index.json') != 0):
                self.loadIndex()                                      # if index alreading exists in file, load from it
            else:
                self.createIndex()                                    # else create index from document files
        self.loadORcreateDocVec()

    def vectorizeDocs(self):                                          # creating vectors of documents
        i = 1
        while(i <= self.numOfDocs):
            self.documentVectors[i] = [0]*len(self.index)             # initializing document vectors
            count = 0
            for word in self.index:
                if count < len(self.index):
                    self.documentVectors[i][count] = self.index[word]['TF-IDFs'][i - 1]  # storing TFxIDF in document vectos
                    count += 1
            i += 1
        self.saveDocVec()

    def saveDocVec(self):                                             # save document vectors to file
        print("\nSaving Document Vectors to json file...\n")
        with open('Submission/files/documentVectors.json', 'w') as file:               # open file as write-mode
            json.dump(self.documentVectors, file, indent=4)           # write dictionary to file
        print("\nDocument Vectors saved successfully!\n")
        file.close()

    def loadDocVec(self):                                             # load document vectors from json
        print("\nLoading Document Vectors from json file...\n")
        with open('Submission/files/documentVectors.json', 'r') as file:               # open file and read-only
            self.documentVectors = json.load(file)                    # load into dictionary from file
        print("Document Vectors loading successful!\n")
        file.close()

    def loadORcreateDocVec(self):
        if(len(self.documentVectors) == 0):                           # if document vectors is empty
            if exists('Submission/files/documentVectors.json') and (os.path.getsize('Submission/files/documentVectors.json') != 0): # check if document vectors file exists and is not empty
                self.loadDocVec()                                     # load document vectors from file 
            else:
                self.vectorizeDocs()                                  # else create document vectors

    def getQuery(self, q):                                            # q = userinput comes from run.py(GUI) 
        q = q.lower()                                                 # casefolding
        q = re.sub(self.punctuations, "", q)                          # removing punctuations
        q_tokens = nltk.word_tokenize(q)                              # tokenize query
        query = [self.lemmatizer.lemmatize(each) for each in q_tokens]  # lemmatizing query terms
        self.QueryIndex(query)

    def QueryIndex(self, query):
        q_index = {}
        for word in query:
            if(word not in self.getStopWordsFromFile()):              # removing stopwords from query
                if(word not in q_index):                              # add distinct words in query-index
                    q_index[word] = {'TFs': 0, 'TF-IDFs': 0}          # making format of the dictionary for every distinct term
                    q_index[word]['TFs'] += 1                         # calculating TF
                else:
                    if(q_index[word]['TFs'] >= 1):
                        q_index[word]['TFs'] += 1
        for word in q_index:
            q_index[word]['TF-IDFs'] = q_index[word]['TFs'] * self.index[word]['IDF']   # calculating TFxIDF of query and terms
        self.vectorizeQuery(q_index)

    def vectorizeQuery(self, q_index):
        queryVector = [0]*len(self.index)                             # creating vector of query
        count = 0
        for word in self.index:
            if (word in q_index):
                queryVector[count] = q_index[word]['TF-IDFs']
            count += 1
        self.cosineSim(queryVector)

    def cosineSim(self, queryVector):                                 # sim(di, q) = DotProd(Di, q) / |Di| * |Q|
        i = 0
        sim = [0]*len(self.documentVectors)                           # calculating cosine similarity between each document and query 
        for doc in self.documentVectors:
            dotProd = self.dotProduct(self.documentVectors[doc], queryVector)
            qMag = self.magnitude(queryVector)
            diMag = self.magnitude(self.documentVectors[doc])
            sim[i] = dotProd/(diMag*qMag)
            i += 1
        self.filteringDocuments(sim)

    def dotProduct(self, di, q):
        dotp = np.dot(di, q)                                          # calculating dotProduct of doc and query
        return (dotp)

    def magnitude(self, d):
        a = np.array(d)
        mag = np.sqrt(a.dot(a))                                       # calculating magnitude
        return (mag)

    def filteringDocuments(self, sim):
        i = 0
        self.result = []
        
        for v in sim:
            if(v >= self.alpha):                                      # filtering documnets with alpha value
                self.result.append(i+1)
            i += 1

    def showResult(self):
        return (self.result)                                          # return the final result to run.py(GUI)
