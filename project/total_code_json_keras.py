import csv
import urllib
from urllib.request import urlopen

import imageio
from bs4 import BeautifulSoup as bs
import requests
import os  # 운영체제와 상호작용을 통한 경로 가져오기
import dlib  # dlib.get_frontal_face_detector(), dlib.shape_predictor(), dlib.load_rgb_image() 사용
import glob  # glob.glob <- 사용자가 제시한 조건에 맞는 파일명을 리스트 형식으로 리턴
import cv2  # opencv
import numpy as np
import json
import math
import pandas as pd
from project.keras_05_pre_train import img_resolution_train
from project.keras_06_pre_test import img_resolution_test
from project.keras_07_train_train import img_resolution_train2
from project.keras_08_train_test import img_resolution_test2

def del_csv():
    df = pd.read_csv('../project/celeba-dataset/list_eval_partition_test.csv')
    df = df.drop([df.index[-1]])
    df.to_csv('../project/celeba-dataset/list_eval_partition_test.csv', index=False)


def Distance(a, b):
    x1, y1, x2, y2 = a[0], a[1], b[0], b[1]
    distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return distance


def Radian(a, b):
    x1, y1, x2, y2 = a[0], a[1], b[0], b[1]
    c = [x1, y2]
    A = Distance(a, b)
    # B = Distance(b, c)
    C = Distance(a, c)
    angle = math.acos(C / A)
    dgrees = math.degrees(angle)
    return int(dgrees)

### 추가 한것 ###
ALL = list(range(0, 68))

# parameter 값으로 자신의 이미지 파일을 저장해 놓은 경로를 입력한다.
def facing_67(face_folder_path, rank):
    predictor_path = "../testing/shape-predict/shape_predictor_68_face_landmarks.dat"
    faces_folder_path = face_folder_path

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    n = 1  # 이미지 번호 증가
    cnt = 0  # 제대로 저장된 이미지 수
    no_dets = 0  # 데이터를 찾지 못한 이미지 수
    minus_value = 0  # 데이터는 있지만 음수가 들어가 있는 이미지 수
    remove_counts = 0

    # dict를 저장하는 리스트
    landmark_dict_list = []

    for f in glob.glob(os.path.join(faces_folder_path, f"{rank}*.jpg")):
        img = dlib.load_rgb_image(f)  # directory 에 있는 이미지 파일 하나씩 불러오기
        r, g, b = cv2.split(img)  # bgr을 rgb 로변경
        cvImg = cv2.merge([b, g, r])
        cvImg = cv2.bilateralFilter(cvImg, 9, 75, 75)

        dets = detector(cvImg, 1)  # face detecting function. 이 결과로 rectangle[] value return
        if len(dets) == 1:  # rectangle[] 에 값이 하나인 경우.

            minus_detector = int(str(dets[0]).replace('[', '').replace(']', '').replace('(', '').replace(')', ''). \
                                 replace(',', '').replace(' ', '').replace('-',
                                                                           '1111111'))  # rectangle() value에 음수가 있는 경우 거르기 위한 과정

            # minus_detector 를 만든 이유는 위에서 dets 가 있는 경우를 걸러도 rectangle() value에 음수가 되는 값이 존재할 수 있기 때문이다.
            # 만일 rectangle() value가 음수가 되면 47번째 'crop = ~' 라인에서 Assertion: _!is_empty() 에러를 반환하는 경우가 발생한다.
            # 구글에서 90개의 이미지를 크롤링한 결과 1장의 rectangle() value가 음수가 되었고 이를 거르는 과정을 설정하였다.

            if minus_detector < 100_000_000_000:  # 위 과정에서 음수가 최소 12자리 되도록 설정 -> xx가 양수이면 1000억 미만
                for _, d in enumerate(dets):
                    shape = predictor(img, d)

                    # 좌표를 저장할 빈 리스트 생성
                    landmark_list = []
                    for i in range(0, shape.num_parts):
                        x = shape.part(i).x
                        y = shape.part(i).y

                        # 좌표값 landmark_list에 저장
                        landmark_list.append([x, y])

                        cv2.putText(cvImg, str(i), (x, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.3, (0, 255, 0))

                    if Radian(landmark_list[27], landmark_list[30]) < 9:
                        # crop = cvImg[d.top():d.bottom(), d.left():d.right()]
                        # cv2.imwrite(f'rich{n}.jpg', crop)  ###@ 여기서 왜 저장 하지????? ### 그리고 이거 실행하면 5개 밖에 사진이 안생김 ### 그리고 저장 폴더 설정은?
                        ### 삭제

                        landmark_dict = dict()
                        for i in ALL:
                            landmark_dict[i] = landmark_list[i]
                        landmark_dict_list.append(landmark_dict)
                        cnt += 1

                    else:
                        os.remove(f)
                        del_csv()
                        remove_counts += 1
                    ### 여기다가 추가 ###
                    with open(f'../testing/json1/rich_{rank}.json', "w") as json_file:
                        json_file.write(json.dumps(landmark_dict_list))
                        json_file.write('\n')

                n += 1

            else:
                os.remove(f)
                del_csv()
                minus_value += 1
                remove_counts += 1
        else:
            os.remove(f)
            del_csv()
            no_dets += 1
            remove_counts += 1


    # 2020. 06. 20. 기준으로 출력한
    # 상위 부자 10명, 각각 9장을 기준으로 correct_return: 82, no_dets: 7, minus_value:1
    # 이미지 크롤링을 진행한 날짜와 시간에 따라 값은 상이할 수 있다.

    # print(f'correct_return: {cnt}')
    # print(f'no_dets: {no_dets}')
    # print(f'minus_value: {minus_value}')
    # print(f'remove counts: {remove_counts}')

img_resolution_train(r"C:\labs\\project\\project\\celeba-dataset")
img_resolution_train2(r"C:\labs\\project\\project\\celeba-dataset\\processed")


url = "https://ceoworld.biz/2020/02/28/rich-list-index-2020/"

res = requests.get(url)
html = res.text.strip()
soup = bs(html, 'html.parser')  # BeautifulSoup -> bs
rich_link = soup.select('tbody.row-hover td.column-2')

rich_list = [str(rich).replace("</td>", '').replace("<td class=\"column-2\">", '') for rich in rich_link]
# print(rich_list)    # 리스트에서 html 코드 제거

rich_friends = [rich.replace(' ', '+') for rich in rich_list]
# print(rich_friends)     # 리스트에서 띄어쓰기 없애고 대신 url 주소에 띄어쓰기로 먹히는 + 대체

r10 = rich_friends[:10]  # 상위 10명만

rank = 101  # 부자 순위대로 나열하기 위한 카운트
for r in r10:
    url = f'https://www.google.com/search?q={r}&sxsrf=ALeKk01WuCtRoFmDGbZmzgJxG5b6wz8VrQ:1592534710712&' \
          'source=lnms&tbm=isch&sa=X&ved=2ahUKEwjEtuWN7ozqAhVbIIgKHcJdD9MQ_AUoAXoECBgQAw&biw=1920&bih=1089'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    html = urlopen(req)
    soup = bs(html, "html.parser")
    img = soup.find_all(class_='t0fcAb')
    n = 1  # 이미지 따로 저장하기 위한 카운트
    for i in img:
        imgUrl = i.attrs['src']
        with urlopen(imgUrl) as f:
            with open('../project/celeba-dataset/imageset_test/' + str(rank) + '_' +  ### 경로 수정 ###
                      r.replace('+', '') + str(n) + '.jpg', mode='wb') as h:  # w - write b - binary
                img = f.read()
                h.write(img)
                with open(os.path.join("../project/celeba-dataset/", 'list_eval_partition_test.csv'), newline='', mode='a') as file:
                    wr = csv.writer(file)
                    files = sorted(glob.glob(os.path.join("../project/celeba-dataset/imageset_test/", '*.jpg')), key=os.path.getctime,
                                   reverse=True)
                    wr.writerow([files[0][40:], 2])


            src = cv2.imread(str(f"../project/celeba-dataset/imageset_test/{str(rank)}_{r.replace('+', '')}{str(n)}.jpg"),
                             cv2.IMREAD_COLOR)
            print(f"../project/celeba-dataset/imageset_test/{str(rank)}_{r.replace('+', '')}{str(n)}.jpg")
            resizing = cv2.resize(src, dsize=(178, 218), interpolation=cv2.INTER_CUBIC)
            gtorgb = cv2.cvtColor(resizing, cv2.COLOR_BGR2RGB)
            blue_mosters = imageio.imwrite(f"../project/celeba-dataset/imageset_test/{str(rank)}_{r.replace('+', '')}{str(n)}.jpg",
                                           im=gtorgb, pilmode='CMYK', as_gray=False)

            img_resolution_test(r"C:\labs\\project\\project\\celeba-dataset")  ### 자기가 화질개선했는데 버리면 안좋아함.....

            img_resolution_test2(r"C:\labs\\project\\project\\celeba-dataset\\processed_test")  ###

            facing_67("../project/celeba-dataset/imageset_test", rank)

        n += 1
        if n > 30:  # 10장만 출력하기
            break
    rank += 1
    print('=========')

    # ../testing/images/101_JeffBezos1.jpg