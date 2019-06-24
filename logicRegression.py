from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
import pandas as pd
import numpy as np
from sklearn import  svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# 逻辑回归分类模型
def myLogicRegression(train_x,train_y,test_x,test_y):
    clf = LogisticRegression(penalty='l2', multi_class='ovr', solver='liblinear')
    # clf = LogisticRegression()
    clf.fit(train_x, train_y)
    score = clf.score(test_x, test_y)
    print("逻辑回归准确率:",score)
def mysvmClassfication(train_x,train_y,test_x,test_y):
    clf = svm.SVC(kernel='linear')
    clf.fit(train_x, train_y)
    score_svm = clf.score(test_x, test_y)
    print("svm准确率:", score_svm)
# 将两个csv文件合并并打乱顺序
def shuffle(data1,data2):
    df1=pd.read_csv(data1)
    df2 = pd.read_csv(data2)
    # sample会打乱dataframe的顺序
    dataframe = pd.concat([df1, df2], axis=0, ignore_index=True).sample(frac=1).reset_index(drop=True)
    return dataframe

# 生成tf-idf 向量
def generate_tf_idf_vec(dataframe_train,dataframe_test):
    fr = open('dictionary_zhnew.txt', 'r', encoding='utf-8')
    dic = {}
    keys = []  # 用来存储读取的顺序
    for line in fr:
        value, key = line.strip().split(':')
        dic[key] = value
        keys.append(key)
    fr.close()
    vectorizer=TfidfVectorizer(binary=False, decode_error='ignore', encoding='utf-8',norm='l2'
                               , token_pattern=r'(?u)\b\w+\b',vocabulary=keys)
    x_train=vectorizer.fit_transform(dataframe_train['content'])
    y_train =(dataframe_train['category'])
    x_test = vectorizer.fit_transform(dataframe_test['content'])
    y_test = dataframe_test['category']
    return x_train,y_train,x_test,y_test

if __name__ =="__main__":
    #1. 将csv文件合并
    dataframe_train = shuffle('1train.csv','0train.csv')
    print('dataframe_train',dataframe_train.shape)
    dataframe_test = shuffle('1test.csv','0test.csv')
    print('dataframe_test',dataframe_test.shape)
    #2.生成tf_idf向量
    train_x, train_y, test_x, test_y =generate_tf_idf_vec(dataframe_train,dataframe_test)
    # 3.训练模型，测试模型 逻辑回归
    myLogicRegression(train_x, train_y, test_x, test_y)
    # 4.模型训练 svm
    mysvmClassfication(train_x, train_y, test_x, test_y)
