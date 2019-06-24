"""将文本整合到 train、test、val 三个文件中"""
"""读取一个文件并转换为一行"""
import os,chardet
import random
import pandas as pd
import numpy as np

# 加载停用词表
stopwords_file = "./stopWords.txt"
stop_f = open(stopwords_file,"r",encoding='utf-8')
stop_words = list()
for line in stop_f.readlines():
    line = line.strip()
    if not len(line):
        continue
    stop_words.append(line)
stop_f.close
print(len(stop_words))
def _read_file(filename):
    with open(filename, 'rb') as file:
        content = file.read()
        codeType = chardet.detect(content)['encoding']
        if codeType.startswith("UTF-8"):
            with open(filename, 'r', encoding='utf-8') as f1:
                return str(f1.read().replace('\n', '').replace('\t', '').replace('\u3000', ''))
        else:
            with open(filename, 'r', encoding='gbk') as f2:
                try:
                    return str(f2.read().replace('\n', '').replace('\t', '').replace('\u3000', ''))
                except UnicodeDecodeError:
                    print(filename)


import jieba
def cut_string(inputstr):
    word_iter = jieba.cut(inputstr)
    word_content = ''
    for word in word_iter:
        word = word.strip(' ')
        if word != '' and word not in stop_words:
            word_content += word + ' '
    return word_content

def save_file(dirname):
    """
    将多个文件整合并存到2个文件中
    dirname: 原数据目录
    文件内容格式:  类别\t内容
    """
    ids =[]
    categorys = []
    contents =[]
    for category in os.listdir(dirname):   # 分类目录
        cat_dir = os.path.join(dirname, category)
        if not os.path.isdir(cat_dir):
            continue
        files = os.listdir(cat_dir)
        random.shuffle(files)
        count = 0
        for cur_file in files:
            filename = os.path.join(cat_dir, cur_file)
            part_titles = cur_file.split("_")
            if(part_titles[0].startswith("DIA")):
                ids.append(part_titles[1])
            else:
                ids.append(part_titles[0])
            content = _read_file(filename)
            word_content = cut_string(content)
            contents.append(word_content)
            categorys.append(category)
    s1 = pd.Series(np.array(categorys[:1850]))
    s2 = pd.Series(np.array(ids[:1850]))
    s3 = pd.Series(np.array(contents[:1850]))
    df_train = pd.DataFrame({"category": s1, "id": s2,"content":s3})
    df_train.to_csv(categorys[0]+"train.csv",encoding='utf-8')
    s4 = pd.Series(np.array(categorys[1850:]))
    s5 = pd.Series(np.array(ids[1850:]))
    s6 = pd.Series(np.array(contents[1850:]))
    df_test = pd.DataFrame({"category": s4, "id": s5, "content": s6})
    df_test.to_csv(categorys[0] + "test.csv",encoding='utf-8')
    print('Finished:', categorys[0])
if __name__ == '__main__':
    # save_file('2ClassificationData/positive')
    save_file('2ClassificationData/negative')
