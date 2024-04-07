'''extracting overlapping users
将对应域的id转成数字 排序 by孙浩然师兄
'''

import pandas as pd
import os
import numpy as np

def read_xinxi(path_s,path_t):
    df1 = pd.read_csv(path_s, delimiter=',', names=['u', 'i', 'r', 't'], engine='python')
    df2 = pd.read_csv(path_t, delimiter=',', names=['u', 'i', 'r', 't'], engine='python')

    print('------------------------------------')
    tip1 = df1.u.nunique()
    tip2 = df2.u.nunique()
    a = df1.i.nunique()
    b = df2.i.nunique()  #df2.i.unique()查看全部
    print(tip1, tip2)
    print(a, b)
    print(df1.shape,df2.shape)

def zhuanshuzi(path_s,path_t):

    datas = pd.read_csv(path_s)
    datat = pd.read_csv(path_t)

    # 用户和物品ID映射字典
    user_mappings = {k: v for v, k in enumerate(datas['reviewerID'].unique())}
    user_mappingt = {k: v for v, k in enumerate(datat['reviewerID'].unique())}

    # 连续数字替换userID和itemID   即重新编号
    datas['reviewerID'] = datas['reviewerID'].map(user_mappings)
    datat['reviewerID'] = datat['reviewerID'].map(user_mappings)

    datas.sort_values(by=['asin'], inplace=True)
    datat.sort_values(by=['asin'], inplace=True)
    item_mappings = {k: v for v, k in enumerate(datas['asin'].unique())}
    item_mappingt = {k: v for v, k in enumerate(datat['asin'].unique())}
    datas['asin'] = datas['asin'].map(item_mappings)
    datat['asin'] = datat['asin'].map(item_mappingt)

    datas.sort_values(by=['reviewerID', 'asin'], inplace=True)
    datat.sort_values(by=['reviewerID', 'asin'], inplace=True)

    #存修改后的数字id的信息
    file_name3 = path_s.replace('_77.csv','_7.csv')
    file_name4 = path_t.replace('_77.csv','_7.csv')

    datas.to_csv(file_name3, index=False, header=False)
    datat.to_csv(file_name4, index=False, header=False)

    return file_name3,file_name4

if __name__ == '__main__':
    cur_path = os.path.abspath(os.path.dirname(__file__))
    #读取overlap user
    file_name1 = cur_path + "\\data\\" + "Digital_Music_77.csv"
    file_name2 = cur_path + "\\data\\" + "Movies_and_TV_77.csv"

    file_name3,file_name4=zhuanshuzi(file_name1,file_name2)

    read_xinxi(file_name3,file_name4)
