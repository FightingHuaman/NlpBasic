import os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA
import jieba
# import pynlpir
import csv
import pandas as pd
import shutil
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import Normalizer
import time
import re
import numpy as np
import csv
import sys

def articleTf_idfMatrix(texts):
    fr = open('dictionary_zhnew.txt', 'r', encoding='utf-8')
    dic = {}
    keys = []  # 用来存储读取的顺序
    for line in fr:
        value, key = line.strip().split(':')
        dic[key] = value
        keys.append(key)
    fr.close()
    # norm='l2',norm范数用于标准化词条向量。None为不归一化  vocabulary=  norm='l2' None
    tfidf_vec = TfidfVectorizer(binary=False, decode_error='ignore', encoding='utf-8', norm='l2',
                                token_pattern=r'(?u)\b\w+\b', vocabulary=keys)
    tfidf_matrix = tfidf_vec.fit_transform(texts)
    return tfidf_matrix

def read_article(category):
    dataframe = pd.read_csv('outfile.csv').dropna(axis=0)
    df2 = dataframe[dataframe['category'] == category]
    articles = df2[['articleid','word_jieba']]
    return  articles

# KMeans Cluster
def kmeansCluster(tfidf_matrix):
    num_clusters = 10
    km_cluster = KMeans(n_clusters=num_clusters, init='k-means++', n_init=200, max_iter=80000,precompute_distances=False)
    results = km_cluster.fit(tfidf_matrix).labels_
    # for label in results:
    #     print(label)

    print('result shape: ', results.shape)
    print('km_cluster.cluster_centers_', km_cluster.cluster_centers_)
    print('km_cluster.cluster_centers_ shape: ', km_cluster.cluster_centers_.shape)
    return results

def move_article(id,indir,outdir):
    titles = os.listdir(indir)
    for title in titles:
        if re.search(str(id),title) is not None:
            shutil.move(indir+title, outdir+title)

if __name__ =="__main__":
    articles = read_article('9999杂类')
    ids = articles['articleid'].tolist()
    texts = articles['word_jieba'].tolist()
    titles = os.listdir("E:/华满_待分组/8/")
    newids = []
    newtexts =[]
    for index,id in enumerate(ids):
        for title in titles:
            if re.search(str(id),title) is not None:
                newtexts.append(texts[index])
                newids.append(id)
    print(len(newtexts),len(newids))
    tfidfMatrix =articleTf_idfMatrix(newtexts)
    results = kmeansCluster(tfidfMatrix)
    movecount =0
    for i in range(len(results)):
        move_article(str(newids[i]),"E:/华满_待分组/8/","E:/聚类文档_new/"+str(results[i])+"/")
        movecount=movecount+1
    print("完成",movecount,"篇文章的移动")