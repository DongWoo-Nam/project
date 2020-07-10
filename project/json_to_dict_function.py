import glob
import json
import csv
import os
import time

import numpy as np
import pandas as pd
from project.class01_Face_Distance_Ratio import FaceDistanceRatio
from project.class01_Face_Distance_Ratio import rich_list

def json_to_dict(path):
    with open(path, mode='r') as file:
        dictionary = json.load(file)
    return dictionary



if __name__ == '__main__':
    print(json_to_dict('../testing/json1/rich_101.json'))
    # fdr.dict_to_df(dictionary, index)
    rich_list = rich_list()
    print(rich_list)


    pt1 = FaceDistanceRatio(json_to_dict('../testing/json1/rich_101.json')[0])
    print(pt1.dict_to_df(rich_list[0]))  # index는 항상 []로 들어와야함 이름을 뽑을 때에도
    print()
    t_start = time.time()
    for i in range(len(rich_list)):
        for j in glob.glob(os.path.join('../testing/json1/', f'rich_{101 + i}.json')):
            with open(j, mode='r') as file:
                lst = json.load(file)
                for k in range(len(lst)):
                    dc = json_to_dict(j)[k]
                    print(dc)
                    pt2 = FaceDistanceRatio(dc)
                    df = pt2.dict_to_df(rich_list[i])
                    print(df)
    t_end = time.time()
    print('경과 시간:', round(t_end-t_start, 4), 'seconds')