#计算文章的相似度
"""
1.已经标记好的文章的内部的相似度的平均值
2.一篇未知文章与所有已知文章的相似度平均值
3.比较，若2>1,则文档与1同类
4.移动文档
"""
import os,re,shutil
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import  numpy as np
import pandas as pd

#文件移动，
# 文件参数：id 输入文件，indir 输入目录，outdir 输出目录
def move_article(id,indir,outdir):
    titles = os.listdir(indir)
    for title in titles:
        if re.search(str(id),title) is not None:
            shutil.move(indir+title, outdir+title)

#   文本相似度，返回一个矩阵
def article_sim(text,text_model):
    fr = open('dictionary_zhnew.txt','r',encoding='utf-8')
    dic = {}
    keys = [] #用来存储读取的顺序
    for line in fr:
        value,key = line.strip().split(':')
        dic[key] = value
        keys.append(key)
    fr.close()
    # norm='l2',norm范数用于标准化词条向量。None为不归一化  vocabulary=  norm='l2' None
    tfidf_vec = TfidfVectorizer(binary=False, decode_error='ignore', encoding='utf-8',norm='l2' , token_pattern=r'(?u)\b\w+\b',vocabulary=keys)
    tfidf_matrix = tfidf_vec.fit_transform(text )
    tfidf_matrix2 = tfidf_vec.fit_transform(text_model)
    tfidf = tfidf_matrix.toarray()
    tf_model = tfidf_matrix2.toarray()
    sim_matrix = np.dot(tfidf, tf_model.T)
    sum= sim_matrix.sum()
    return  sim_matrix

def read_article(category):
    dataframe = pd.read_csv('myoutfile.csv').dropna(axis=0)
    df2 = dataframe[dataframe['category'] == category]
    articles = df2[['id','word_jieba']]
    return  articles

def get_model(id):
    dataframe = pd.read_csv('outfile.csv').dropna(axis=0)
    df = dataframe.astype(str)
    model_df = df[df['articleid'] == id]
    model = model_df['word_jieba']
    return model


if __name__ =="__main__":
    # 1.计算文本内部相似度的平均值
    articles = read_article("产品类_客服邮件")
    texts = articles["word_jieba"]
    lenth = len(texts)
    sim_matrix = article_sim(texts,texts)
    sim_inside = ((sim_matrix.sum()-articles.size)/(lenth * lenth - lenth))
    print(sim_inside)
    dirname = 'C:/Users/Administrator/Desktop/8/'
    ids = []
    for txtname in os.listdir(dirname):  # 文件名
        data = txtname.split('_')
        id = data[1]
        ids.append(id)
        txtCompareCount =0
        txtMove =0
    for id in ids:
        text1 = get_model(str(id))
        sim_matrix2 = article_sim(texts,text1)
        sim_outside = sim_matrix2.mean()
        txtCompareCount+=1
        print("已经比较",txtCompareCount,"篇文本")
        if sim_outside >=sim_inside:
            print(sim_outside)
            txtMove+=1
            # 产品类_客服询盘邮件 产品类_卖家运营
            move_article(id,dirname,"C:/Users/Administrator/Desktop/D_3_luo - 副本/产品类_客服询盘邮件/")
    print("已经移动", txtMove, "篇文本")