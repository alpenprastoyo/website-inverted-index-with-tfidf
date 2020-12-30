import os
import string
import re
from nltk.stem import *
import nltk
import itertools
import math
import operator
import sys
from statistics import mean
import json
from bs4 import BeautifulSoup
from flask import Flask, render_template, request
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
'''
This script is the main program of this project, by running it the user is displayed a GUI where he can choose some
settings and start running his queries and seeing the results
'''

def tokenize_and_remove_punctuations(s):
    s = BeautifulSoup(s, "lxml").text
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    s   = stemmer.stem(s)
    translator = str.maketrans('','',string.punctuation)
    modified_string = s.translate(translator)
    modified_string = ''.join([i for i in modified_string if not i.isdigit()])
    return nltk.word_tokenize(modified_string)

def get_stopwords():
    factory = StopWordRemoverFactory()
    stop_words = factory.get_stop_words()
    return stop_words

def stem_words(tokens):
    # print(tokens)
    # stemmer = PorterStemmer()
    # stemmed_words = [stemmer.stem(token) for token in tokens]
    # print(stemmed_words)
    # sys.exit()
    return tokens

def remove_stop_words(tokens):
    stop_words = get_stopwords()
    filtered_words = [token for token in tokens if token not in stop_words and len(token) > 2]
    return filtered_words



def calculate_tf(tokens):
    tf_score = {}
    for token in tokens:
        tf_score[token] = tokens.count(token)
    return tf_score

def get_vocabulary(data):
    tokens = []
    for token_list in data.values():
        tokens = tokens + token_list
    fdist = nltk.FreqDist(tokens)
    return list(fdist.keys())

def preprocess_data(contents):
    dataDict = {}
    for content in contents:
        tokens = tokenize_and_remove_punctuations(content[1])
        filtered_tokens = remove_stop_words(tokens)
        stemmed_tokens = stem_words(filtered_tokens)
        filtered_tokens1 = remove_stop_words(stemmed_tokens)
        dataDict[content[0]] = filtered_tokens1
    return dataDict

def calculate_idf(data):
    idf_score = {}
    N = len(data)
    all_words = get_vocabulary(data)
    for word in all_words:
        word_count = 0
        for token_list in data.values():
            if word in token_list:
                word_count += 1
        idf_score[word] = math.log10(N/word_count)
    return idf_score

def calculate_tfidf(data, idf_score):
    scores = {}
    for key,value in data.items():
        scores[key] = calculate_tf(value)
    for doc,tf_scores in scores.items():
        for token, score in tf_scores.items():
            tf = score
            idf = idf_score[token]
            tf_scores[token] = tf * idf
    return scores

def preprocess_queries(path):
    queriesDict = {}
    #queries = open(path,'r').read().split('\n')
    queries = path.split('\n')
    i = 1
    for query in queries:
        tokens = tokenize_and_remove_punctuations(query)
        filtered_tokens = remove_stop_words(tokens)
        stemmed_tokens = stem_words(filtered_tokens)
        filtered_tokens1 = remove_stop_words(stemmed_tokens)
        queriesDict[i] = filtered_tokens1
        i+=1
    return queriesDict

def calculate_tfidf_queries(queries, idf_score):
    scores = {}
    for key, value in queries.items():
        scores[key] = calculate_tf(value)
    for key, tf_scores in scores.items():
        for token, score in tf_scores.items():
            idf = 0
            tf = score
            if token in idf_score.keys():
                idf = idf_score[token]
            tf_scores[token] = tf * idf
    return scores

def generate_inverted_index(data):
    all_words = get_vocabulary(data)
    index = {}
    for word in all_words:
        for doc, tokens in data.items():
            if word in tokens :
                if word in index.keys():
                    index[word].append(doc)
                else:
                    index[word] = [doc]
    return index

def get_relevance(path):
    relevances = {}
    data = open(path,'r').read()
    for line in data.split('\n'):
        tokens = line.split(" ")
        if int(tokens[0]) in relevances.keys():
            relevances[int(tokens[0])].append(int(tokens[1]))
        else:
            relevances[int(tokens[0])] = [int(tokens[1])]
    return relevances

def find_precision_recall(relevances, docList):
    relevant_docs = len([doc for doc in docList if doc in relevances])
    total_relevant = len(relevances)
    total_docs = len(docList)
    precision = relevant_docs/total_docs
    recall = relevant_docs/total_relevant
    return precision, recall


#Querry
def query(search):
    with open('4. idf_scores.json') as json_file:
        idf_scores = json.load(json_file)
    with open('3. inverted_index.json') as json_file:
        inverted_index = json.load(json_file)
    with open('5. scores.json') as json_file:
        scores = json.load(json_file)
    with open('1. data.json') as json_file:
        data = json.load(json_file)
    
    

    queries = preprocess_queries(search)
    with open('6. queries.json', 'w') as outfile:
        json.dump(queries, outfile)

    #Skor Querry
    query_scores = calculate_tfidf_queries(queries,idf_scores)
    with open('7. query_scores.json', 'w') as outfile:
        json.dump(query_scores, outfile)
    query_docs = {}
    for key, value in queries.items():
        doc_sim = {}
        for term in value:
            if term in inverted_index.keys():
                docs = inverted_index[term]
                for doc in docs:
                    # print(doc)
                    # print(scores[str(doc)])
                    doc_score = scores[str(doc)][term]
                    doc_length = math.sqrt(sum(x ** 2 for x in scores[str(doc)].values()))
                    query_score = query_scores[key][term]
                    query_length = math.sqrt(sum(x ** 2 for x in query_scores[key].values()))
                    cosine_sim = (doc_score * query_score) / (doc_length * query_length)
                    if doc in doc_sim.keys():
                        doc_sim[doc] += cosine_sim
                    else:
                        doc_sim[doc] = cosine_sim
        ranked = sorted(doc_sim.items(), key=operator.itemgetter(1), reverse=True)
        query_docs[key] = ranked
    with open('9. query_docs.json', 'w') as outfile:
        json.dump(query_docs, outfile)
    page_rank = []    
    for rank in ranked:
        rank = list(rank)
        rank += [data[rank[0]][1],data[rank[0]][2]]
        page_rank += [rank]
        # print(rank)
    # print(page_rank)
    with open('10. page_rank.json', 'w') as outfile:
        json.dump(page_rank, outfile)
    return page_rank

# data = query('universitas sebelas maret')

app = Flask(__name__)

@app.route('/')
def hello():
    return render_template('search.html')
@app.route('/inverted', methods = ['POST'])
def inverted():
    data = query(request.form['search'])
    return render_template('result.html', data = data, len = len(data) )



app.run(debug=True)





