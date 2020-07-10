import glob
import json
import os
import time
import pickle

import numpy as np
import pandas as pd
from project.class01_Face_Distance_Ratio import FaceDistanceRatio
from project.class01_Face_Distance_Ratio import rich_list
from project.json_to_dict_function import json_to_dict
from project.test_img_total_code_json_keras import test_image_df_recall


def rich_total_dataframe():
    t_start = time.time()
    df = pd.DataFrame(columns=['right_eye_width', 'right_eye_height', 'right_eye_shape', 'right_eyebrow',
                               'left_eye_width', 'left_eye_height', 'left_eye_shape', 'left_eyebrow',
                               'eye_between', 'nose_width', 'nose_height', 'philtrum',
                               'mouth_width', 'mouth_height', 'mouth_shape', 'jaw_height'])
    rich_list = []
    with open('rich_list.csv', mode='r') as file:
        for line in file:
            rich = line.strip().split(',')
            rich_list.append(rich)

    for i in range(101, 600):
        if i == 227:
            pass
        elif i == 404:
            pass
        elif i == 575:
            pass
        else:
            with open(f'../project/dataframe/{i}_{rich_list[0][i - 101]}.pkl', mode='rb') as f:
                p = pickle.load(f)
                df2 = pd.DataFrame(p)
                sr = df2.mean(axis=0)
                # print(i, rich_list[i - 101])  # 201, 523, 544
                # print(sr)
                df.loc[i - 101] = sr
                df.rename(index={(i - 101): rich_list[0][i - 101]}, inplace=True)
    # print(df)
    # print(len(df))
    sr = df.mean(axis=0)
    df.loc['rich_mean'] = sr

    sr_cv = df.std(axis=0) / df.mean(axis=0)
    df.loc['rich_cv'] = sr_cv
    # print(df)
    #
    # print(df.T.sort_values('rich_cv'))
    # print(df.T.sort_values('rich_cv').iloc[1:11])
    df_type = df.T.sort_values('rich_cv').iloc[1:11].T  # 여기다가 추가
    # 0 자수성가형, 1 금수저형, 2 투자의귀재, 3 또라이형, 4 자퇴형
    # 5 결혼형, 6 시인형, 7 UN특사형, 8 정치인형, 9 professor type
    type = [0, 4, 4, 0, 4, 1, 0, 0, 0, 4, 1, 0, 0, 1, 1, 1, 5, 1, 5, 3,
            0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 4, 0, 1, 0, 0, 1, 1, 5, 0, 0,
            1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 5, 1, 0, 1, 1, 0, 1, 0, 0, 8,
            0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 2, 1, 2, 0, 1, 0, 1, 2,
            2, 0, 0, 0, 0, 0, 1, 1, 0, 1, 2, 0, 1, 1, 1, 1, 4, 2, 0, 0,
            0, 0, 2, 0, 5, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 4,
            0, 1, 2, 2, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 1, 4, 1, 0, 0, 4, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 2, 0,
            0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0,
            1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 5, 0, 0, 1, 0, 0,
            1, 0, 1, 0, 4, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1, 1, 2, 1, 2, 1, 0, 0,
            1, 1, 2, 0, 0, 1, 0, 1, 0, 2, 2, 4, 0, 5, 0, 0, 0, 0, 0, 0,
            0, 2, 0, 0, 1, 0, 5, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1,
            0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
            0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 2, 1, 1, 1, 2, 0, 1, 2, 0, 0,
            0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 2, 1, 0, 7, 1, 1, 0, 1, 1, 1,
            1, 0, 0, 1, 1, 1, 0, 2, 2, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1,
            1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 3, 1, 0, 3, 1, 1,
            1, 0, 1, 1, 0, 0, 0, 3, 0, 0, 1, 1, 1, 1, 6, 0, 0, 1, 1, 0,
            1, 1, 0, 0, 1, 2, 0, 1, 0, 0, 0, 0, 0, 1, 9, 1, 2, 2, 1, 1,
            1, 0, 5, 0, 0, 4, 5, 0, 0, 1, 4, 3, 0, 4, 1, 1, 0, 0, 1, 3,
            0, 0, 2, 2, 2, 0, 1, 1, 0, 4, 0, 1, 1, 1, 1, 1, 0, 1, 1, 3,
            5, 0, 0, 0, 1, 1, 1, 0, 1, 5, 1, 0, 0, 6, 1, 4, 0, 2, 0, 1,
            0, 0, 1, 1, 3, 0, 5, 1, 0, 0, 1, 1, 1, 1, 0, 1, None, None]
    df_type['type'] = type

    t_end = time.time()
    print(f'경과 시간 : {round(t_end - t_start, 4)} seconds')

    return df_type


def return_type(name_number):
    # [8, 4, 0, 7, 9, 3, 10, 15, 12, 11]
    new_df = test_image_df_recall(name_number)[
        ['eye_between', 'left_eye_width', 'right_eye_width', 'left_eyebrow', 'nose_width',
         'right_eyebrow', 'nose_height', 'jaw_height', 'mouth_width', 'philtrum']]
    new_df = new_df.values.tolist()[0]

    data = rich_total_dataframe()
    blist = []

    for x in range(496):
        slist = []
        for y in range(10):
            slist.append(data.iloc[x][y])
        blist.append(slist)

    clist = []
    for i in range(496):
        sum = 0
        for j in range(10):
            sum += abs(blist[i][j] - new_df[j])
        clist.append(sum)

    # return data.index[np.argmin(clist)]     # 부자이름
    # 0 자수성가형, 1 금수저형, 2 투자의귀재, 3 또라이형, 4 자퇴형
    # 5 결혼형, 6 시인형, 7 UN특사형, 8 정치인형, 9 professor type
    # print(data.index[np.argmin(clist)])
    t = int(data.iloc[np.argmin(clist), 10])
    print(f'당신과 가장 닮은 부자는 {data.index[np.argmin(clist)]}입니다')
    print(t)
    if t == 0:
        return('자수성가형!\n뼈빠지게 고생하는 타입인 당신!\n언젠간 포브스 500에 당신의 이름이 실릴 날이...\n(ex. 제프 베조스)')
    elif t == 1:
        return('금수저형!\n꽝! 다음 생애에... (ex. 이건희)')
    elif t == 2:
        return('투자의귀재!\n보는 눈이 있는 당신!\n손대는 것마다 투자성공!\n(ex. 버핏)')
    elif t == 3:
        return('또라이형!\n 모 아니면 도! 포브스와 정신병원의 갈림길에 서 있는 당신!\n(ex. 일론머스크)')
    elif t == 4:
        return('자퇴형!\n 일단 자퇴를 해라. \n그러면 성공할 것이다.\n(ex. 주커버그, 빌게이츠, ...)')
    elif t == 5:
        return('결혼형!\n 배우자 복이 있는 당신!\n 행복하세요! (ex. 맥킨지 베조스)')
    elif t == 6:
        return('시인형!\n 당신은 부자의 관상은 아닙니다.. 하지만!\n17세기 최고의 시인이 될 수 있는 관상!\n돈보다 문학을 선택한 당신! 화이팅! (ex. 월터스콧)')
    elif t == 7:
        return('UN특사형!\n 당신은 부자의 관상은 아닙니다.. 하지만!\n국제무대의 비둘기가 될 수 있는 관상!\nun특사에 도전해보는 것은 어떨까요?! (ex. Peter Thomson)')
    elif t == 8:
        return('정치인형!\n 당신은 부자의 관상은 아닙니다.. 하지만!\n뒷주머니로 챙기기 나름!\n 정치로 세계 500대 부자에 도전해 보세요! \n(ex.Elaine Marshall)')
    elif t == 9:
        return('professor type!\n 당신은 부자의 관상이 아닙니다..\n 하지만! 대학원 5년을 견디면...\n 교수가 될, 수도 있는 당신! 화이팅..\n (ex. Mary Malone)')




if __name__ == '__main__':
    # print(json_to_dict('../testing/json1/rich_101.json'))
    # fdr.dict_to_df(dictionary, index)
    # rich_list = rich_list()
    # print(rich_list)

    # JSON파일을 DF으로 변환하여 pickle에 저장
    # t_start = time.time()
    # for i in range(len(rich_list)):
    #     df = pd.DataFrame(columns=['right_eye_width', 'right_eye_height', 'right_eye_shape', 'right_eyebrow',
    #                                'left_eye_width', 'left_eye_height', 'left_eye_shape', 'left_eyebrow',
    #                                'eye_between', 'nose_width', 'nose_height', 'philtrum',
    #                                'mouth_width', 'mouth_height', 'mouth_shape', 'jaw_height'])
    #     if i <= 4:
    #         for j in glob.glob(os.path.join('../testing/json1/', f'rich_{101 + i}.json')):
    #             with open(j, mode='r') as file:
    #                 lst = json.load(file)
    #                 for k in range(len(lst)):
    #                     dc = json_to_dict(j)[k]
    #                     pt2 = FaceDistanceRatio(dc)
    #                     data = pt2.dict_to_series()
    #                     df.loc[k] = data
    #                 print(df)
    #                 with open(f'../project/dataframe/{101 + i}_{rich_list[i].strip()}.pkl', mode='wb') as pic:
    #                     pickle.dump(df, pic)
    #     elif i == 5:
    #         for j in glob.glob(os.path.join('../testing/json1/', 'rich_599.json')):
    #             with open(j, mode='r') as file:
    #                 lst = json.load(file)
    #                 for k in range(len(lst)):
    #                     dc = json_to_dict(j)[k]
    #                     pt2 = FaceDistanceRatio(dc)
    #                     data = pt2.dict_to_series()
    #                     df.loc[k] = data
    #                 print(df)
    #                 with open(f'../project/dataframe/106_{rich_list[i].strip()}.pkl', mode='wb') as pic:
    #                     pickle.dump(df, pic)
    #     else:
    #         for j in glob.glob(os.path.join('../testing/json1/', f'rich_{100 + i}.json')):
    #             with open(j, mode='r') as file:
    #                 lst = json.load(file)
    #                 for k in range(len(lst)):
    #                     dc = json_to_dict(j)[k]
    #                     pt2 = FaceDistanceRatio(dc)
    #                     data = pt2.dict_to_series()
    #                     df.loc[k] = data
    #                 print(df)
    #                 with open(f'../project/dataframe/{101 + i}_{rich_list[i].strip()}.pkl', mode='wb') as pic:
    #                     pickle.dump(df, pic)
    # t_end = time.time()
    # print('경과 시간:', round(t_end-t_start, 4), 'seconds')



    # for i in (201, 523, 544):
    #     df = pd.DataFrame(columns=['right_eye_width', 'right_eye_height', 'right_eye_shape', 'right_eyebrow',
    #                                'left_eye_width', 'left_eye_height', 'left_eye_shape', 'left_eyebrow',
    #                                'eye_between', 'nose_width', 'nose_height', 'philtrum',
    #                                'mouth_width', 'mouth_height', 'mouth_shape', 'jaw_height'])
    #     for j in glob.glob(os.path.join('../testing/json1/', f'rich_{i}.json')):
    #         with open(j, mode='r') as file:
    #             lst = json.load(file)
    #             for k in range(len(lst)):
    #                 dc = json_to_dict(j)[k]
    #                 pt2 = FaceDistanceRatio(dc)
    #                 data = pt2.dict_to_series()
    #                 df.loc[k] = data
    #             print(df)
    #             with open(f'../project/dataframe/{i}_{rich_list[i - 101].strip()}.pkl', mode='wb') as pic:
    #                 pickle.dump(df, pic)
    #
    # with open(f'../project/dataframe/201_{rich_list[100]}.pkl', mode='rb') as f:
    #     p = pickle.load(f)
    #     df2 = pd.DataFrame(p)
    #     print(df2)
    # with open(f'../project/dataframe/523_Agnete Thinggaard.pkl', mode='rb') as f:
    #     p = pickle.load(f)
    #     df2 = pd.DataFrame(p)
    #     print(df2)
    # with open(f'../project/dataframe/544_Ted Lerner.pkl', mode='rb') as f:
    #     p = pickle.load(f)
    #     df2 = pd.DataFrame(p)
    #     print(df2)

    print(rich_total_dataframe())
    return_type(1006)
