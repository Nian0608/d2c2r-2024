'''extracting overlapping users
将数据集划分train test 并只保留u i r
'''

import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split

def read_xinxi(path_s,path_t):
    df1 = pd.read_csv(filepath_or_buffer=path_s, sep=',', names=['u', 'i', 'r', 't'], engine='python')
    df2 = pd.read_csv(filepath_or_buffer=path_t, sep=',', names=['u', 'i', 'r', 't'], engine='python')

    print('------------------------------------')
    tip1 = df1.u.nunique()
    tip2 = df2.u.nunique()
    a = df1.i.nunique()
    b = df2.i.nunique()  #df2.i.unique()查看全部
    print(tip1, tip2)
    print(a, b)
    print(df1.shape,df2.shape)

def chouqu(path_s,path_t):
    df1 = pd.read_csv(filepath_or_buffer=path_s, sep=',', names=['u', 'i', 'r', 't'], engine='python')
    df2 = pd.read_csv(filepath_or_buffer=path_t, sep=',', names=['u', 'i', 'r', 't'], engine='python')
    tip1 = df1.u.nunique()
    tip2 = df2.u.nunique()
    list1=range(round(tip1*0.5)-1,round(tip1*0.7))  #生成test集合的uid
    list2=range(round(tip1 * 0.7), round(tip1* 0.9))
    print(list1,list2)
    set_test1=set(list1)    #将test集合uid转为set
    set_test2 = set(list2)
    print(set_test1)
    print(set_test2)
    train1 = df1[~df1['u'].isin(set_test1)]   #train删除testuid
    train2 = df2[~df2['u'].isin(set_test2)]

    testvalid1 = df1[df1['u'].isin(set_test1)]    #test即为testuid
    testvalid2 = df2[df2['u'].isin(set_test2)]
    print(train2, testvalid2)
    train1 = train1.drop(['t'], axis=1)   #train删除时间列
    train2 = train2.drop(['t'], axis=1)

    testvalid1 = testvalid1.drop(['r','t'], axis=1)   #train删除评分列和时间列
    testvalid2 = testvalid2.drop(['r', 't'], axis=1)

    u1_valid, u1_test, i1_valid, i1_test = \
        train_test_split(testvalid1['u'], testvalid1['i'], test_size=0.50, random_state=0)
    valid1=pd.DataFrame()
    valid1['u']=u1_valid
    valid1['i'] = i1_valid
    valid1.sort_values(by=['u', 'i'], inplace=True)
    test1 = pd.DataFrame()
    test1['u'] = u1_test
    test1['i'] = i1_test
    test1.sort_values(by=['u', 'i'], inplace=True)

    u2_valid, u2_test, i2_valid, i2_test = \
        train_test_split(testvalid2['u'], testvalid2['i'], test_size=0.50, random_state=0)
    valid2 = pd.DataFrame()
    valid2['u'] = u2_valid
    valid2['i'] = i2_valid
    valid2.sort_values(by=['u', 'i'], inplace=True)
    test2 = pd.DataFrame()
    test2['u'] = u2_test
    test2['i'] = i2_test
    test2.sort_values(by=['u', 'i'], inplace=True)

    #print(path_s,path_t)
    #修改保存文件夹路径名称
    path_s_train = path_s.replace('\data_pre\data', '\dataset_2\Phones_Ele')\
        .replace('Cell_Phones_and_Accessories_6', 'train')
    path_s_valid = path_s.replace('\data_pre\data', '\dataset_2\Phones_Ele') \
        .replace('Cell_Phones_and_Accessories_6', 'valid')
    path_s_test= path_s.replace('\data_pre\data', '\dataset_2\Phones_Ele')\
        .replace('Cell_Phones_and_Accessories_6', 'test')

    path_t_train = path_t.replace('\data_pre\data', '\dataset_2\Ele_Phones')\
        .replace('Electronics_6', 'train')
    path_t_valid = path_t.replace('\data_pre\data', '\dataset_2\Ele_Phones') \
        .replace('Electronics_6', 'valid')
    path_t_test = path_t.replace('\data_pre\data', '\dataset_2\Ele_Phones')\
        .replace('Electronics_6','test')

    train1.to_csv(path_s_train, index=False, header=False)
    test1.to_csv(path_s_test, index=False, header=False)
    train2.to_csv(path_t_train, index=False, header=False)
    test2.to_csv(path_t_test, index=False, header=False)

    valid1.to_csv(path_s_valid, index=False, header=False)
    valid2.to_csv(path_t_valid, index=False, header=False)

if __name__ == '__main__':
    cur_path = os.path.abspath(os.path.dirname(__file__))
    #修改初始文件夹名称
    name1='Cell_Phones_and_Accessories_6.csv'
    name2='Electronics_6.csv'
    #读取overlap user
    file_name1 = cur_path + '\\data\\' + name1
    file_name2 = cur_path + '\\data\\' + name2
    chouqu(file_name1, file_name2)


