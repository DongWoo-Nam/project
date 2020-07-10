import csv
import urllib
from urllib.request import urlopen
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
    predictor_path = "../testing/shape-predictor"
    faces_folder_path = face_folder_path

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    n = 1  # 이미지 번호 증가
    no_dets = 0  # 데이터를 찾지 못한 이미지 수
    minus_value = 0  # 데이터는 있지만 음수가 들어가 있는 이미지 수
    remove_counts = 0

    for f in glob.glob(os.path.join(faces_folder_path, f"{rank}*.jpg")):
        img = dlib.load_rgb_image(f)
        r, g, b = cv2.split(img)
        cvImg = cv2.merge([b, g, r])
        cvImg = cv2.bilateralFilter(cvImg, 9, 75, 75)

        dets = detector(cvImg, 1)
        if len(dets) == 1:

            minus_detector = int(str(dets[0]).replace('[', '').replace(']', '').replace('(', '').replace(')', ''). \
                                 replace(',', '').replace(' ', '').replace('-',
                                                                           '1111111'))
            if minus_detector < 100_000_000_000:
                for _, d in enumerate(dets):
                    shape = predictor(img, d)

                    for i in range(0, shape.num_parts):
                        crop = cvImg[d.top():d.bottom(), d.left():d.right()]
                        cv2.imwrite(f'{rank}rich{n}.jpg', crop)

                    else:
                        remove_counts += 1

                n += 1

            else:
                minus_value += 1
                remove_counts += 1
        else:
            no_dets += 1
            remove_counts += 1

for rank in range(101, 600):
    facing_67('../test2/imageset_test/', rank)