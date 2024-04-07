import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split

def chuli(path_s,path_t):
    train_s_path=path_s+'\\train.csv'
    train_t_path = path_t + '\\train.csv'
    valid_s_path = path_s + '\\valid.csv'
    valid_t_path = path_t + '\\valid.csv'
    test_s_path = path_s + '\\test.csv'
    test_t_path = path_t + '\\test.csv'

    dfs1 = pd.read_csv(filepath_or_buffer=train_s_path, sep=',', names=['u', 'i', 'r'], engine='python')
    dfs2 = pd.read_csv(filepath_or_buffer=valid_s_path, sep=',', names=['u', 'i'], engine='python')
    dfs3 = pd.read_csv(filepath_or_buffer=test_s_path, sep=',', names=['u', 'i'], engine='python')

    dft1 = pd.read_csv(filepath_or_buffer=train_t_path, sep=',', names=['u', 'i', 'r'], engine='python')
    dft2 = pd.read_csv(filepath_or_buffer=valid_t_path, sep=',', names=['u', 'i'], engine='python')
    dft3 = pd.read_csv(filepath_or_buffer=test_t_path, sep=',', names=['u', 'i'], engine='python')

    duplicate_values = dfs1['u'].value_counts()
    duplicate_values_gt_10 = duplicate_values[(duplicate_values > 40) ] #(duplicate_values > 10) & (duplicate_values <= 20)
    set1 = set(duplicate_values_gt_10.index)

    dft2_valid = dft2[dft2['u'].isin(set1)]
    dft3_test = dft3[dft3['u'].isin(set1)]

    print(set1)
    print(dft2_valid)
    print(dft3_test)

    path_t_valid = valid_t_path.replace('valid', 'valid_50')
    path_t_test = test_t_path.replace('test', 'test_50')
    dft2_valid.to_csv(path_t_valid, index=False, header=False)
    dft3_test.to_csv(path_t_test, index=False, header=False)

if __name__ == '__main__':
    cur_path = os.path.abspath(os.path.dirname(__file__))
    cur_path=cur_path.replace('\data_pre', '')
    #修改初始文件夹名称
    name1='Phones_Ele'
    name2='Ele_Phones'
    #读取overlap user
    file_name1 = cur_path + '\\dataset\\' + name1
    file_name2 = cur_path + '\\dataset\\' + name2
    chuli(file_name1, file_name2)

