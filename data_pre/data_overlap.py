'''extracting overlapping users
http://jmcauley.ucsd.edu/data/amazon/links.html
分别提取两个域中重叠用户ID及其对应的数据 by常艺茹师姐
'''
import pandas as pd
import numpy as np
import gzip, os

def parse(path):
    with gzip.open(path, 'rt', encoding='utf-8') as file:
        for line in file:
            # 将true和false替换为小写形式
            line = line.replace("true", "True").replace("false", "False")
            yield eval(line)

def getDF(path):
    df = {}
    for i,d in enumerate(parse(path)):
        df[i] = d

    return pd.DataFrame.from_dict(df, orient='index')

def construct(path_s, path_t):

    s_5core, t_5core = getDF(path_s), getDF(path_t)
    s_users = set(s_5core['reviewerID'].tolist())
    t_users = set(t_5core['reviewerID'].tolist())
    overlapping_users = s_users & t_users

    s = s_5core[s_5core['reviewerID'].isin(overlapping_users)][['reviewerID','asin','overall','unixReviewTime']]
    t = t_5core[t_5core['reviewerID'].isin(overlapping_users)][['reviewerID','asin','overall','unixReviewTime']]

    #存储转为csv
    csv_path_s = path_s.replace('reviews_','').replace('_5.json.gz','_77.csv')
    csv_path_t = path_t.replace('reviews_','').replace('_5.json.gz','_77.csv')
    s.to_csv(csv_path_s, index=False)
    t.to_csv(csv_path_t, index=False)

    print('Build raw data to %s.' % csv_path_s)
    print('Build raw data to %s.' % csv_path_t)


if __name__ == '__main__':
    cur_path = os.path.abspath(os.path.dirname(__file__))
    #读>=5的json
    path_s=cur_path +"\\data\\reviews_Digital_Music_5.json.gz"
    path_t=cur_path +"\\data\\reviews_Movies_and_TV_5.json.gz"
    construct(path_s, path_t)
