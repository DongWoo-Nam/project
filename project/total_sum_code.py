import csv
import pickle
import time
import imageio
import os  # 운영체제와 상호작용을 통한 경로 가져오기
import dlib  # dlib.get_frontal_face_detector(), dlib.shape_predictor(), dlib.load_rgb_image() 사용
import glob  # glob.glob <- 사용자가 제시한 조건에 맞는 파일명을 리스트 형식으로 리턴
import cv2  # opencv
import numpy as np
import json
import math
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import pyramid_reduce
import warnings
warnings.filterwarnings(action='ignore')
from math import pi
from project.class01_Face_Distance_Ratio import FaceDistanceRatio
from project.json_to_dict_function import json_to_dict
from keras.layers import Conv2D, Input, Activation
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from skimage.transform import pyramid_expand
from project.keras_01_Subpixel import Subpixel
from project.keras_02_Datagenerator import DataGenerator
import matplotlib.font_manager as fm






# base_path = r"C:\labs\\project\\project\\celeba-dataset"
def img_resolution_train(base_path):
    img_base_path = os.path.join(base_path, "img_align_celeba")
    target_img_path = os.path.join(base_path, "processed")

    eval_list = np.loadtxt(os.path.join(base_path, "list_eval_partition.csv"),
                           dtype=str, delimiter=',', skiprows=1)

    img_sample = cv2.imread(os.path.join(img_base_path, eval_list[0][0]))

    h, w, _ = img_sample.shape
    print(h, w)
    # 이미지로 crop
    crop_sample = img_sample[int((h-w)/2):int(-(h-w)/2), :]

    # 이미지 4배 축소 후 normalize
    resized_sample = pyramid_reduce(crop_sample, downscale=4, multichannel=True) # 컬러채널 허용

    pad = int((crop_sample.shape[0] - resized_sample.shape[0]) / 2)

    padded_sample = cv2.copyMakeBorder(resized_sample, top=pad + 1, bottom=pad,
                                       left=pad + 1, right=pad, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))

    print(crop_sample.shape, padded_sample.shape)

    # fig, ax = plt.subplots(1, 4, figsize=(12, 5))
    # ax = ax.ravel()
    # ax[0].imshow(img_sample)
    # ax[1].imshow(crop_sample)
    # ax[2].imshow(resized_sample)
    # ax[2].imshow(cv2.resize(resized_sample, dsize=(45, 45)))
    # ax[3].imshow(padded_sample)
    # plt.show()

    downscale = 4

    # 이미지 train 할 파일 생성
    for i, e in enumerate(eval_list):
        if i == (len(eval_list)-1):
            break
        filename, ext = os.path.splitext(e[0])
        img_path = os.path.join(img_base_path, e[0])
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        crop = img[int((h - w) / 2):int(-(h - w) / 2), :]
        crop = cv2.resize(crop, dsize=(176, 176))
        resized = pyramid_reduce(crop, downscale=downscale, multichannel=True)  # 컬러 채널 허용
        norm = cv2.normalize(crop.astype(np.float64), None, 0, 1, cv2.NORM_MINMAX)

        if int(e[1]) == 0:  # Train
            np.save(os.path.join(target_img_path, "x_train", filename + ".npy"), resized)
            np.save(os.path.join(target_img_path, "y_train", filename + ".npy"), norm)
        else:  # Validation
            np.save(os.path.join(target_img_path, "x_val", filename + ".npy"), resized)
            np.save(os.path.join(target_img_path, "y_val", filename + ".npy"), norm)

    return None

# base_path = r"C:\labs\\project\\project\\celeba-dataset\\processed"
def img_resolution_train2(base_path):
    # train, validation, test 파일 list 만들기
    x_train_list = sorted(glob.glob(os.path.join(base_path, 'x_train', '*.npy')))
    x_val_list = sorted(glob.glob(os.path.join(base_path, 'x_val', '*.npy')))

    print(len(x_train_list), len(x_val_list))
    print(x_train_list[0])


    x1 = np.load(x_train_list[0])
    x2 = np.load(x_val_list[0])

    print(x1.shape, x2.shape)

    # plt.subplot(1, 2, 1)
    # plt.imshow(x1)
    # plt.subplot(1, 2, 2)
    # plt.imshow(x2)
    # plt.show()

    train_gen = DataGenerator(list_IDs=x_train_list, labels=None, batch_size=16, dim=(44,44),
                              n_channels=3, n_classes=None, shuffle=True)

    val_gen = DataGenerator(list_IDs=x_val_list, labels=None, batch_size=16, dim=(44,44),
                            n_channels=3, n_classes=None, shuffle=False)

    upscale_factor = 4
    inputs = Input(shape=(44, 44, 3))

    net = Conv2D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu')(inputs)
    net = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')(net)
    net = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(net)
    net = Conv2D(filters=upscale_factor**2, kernel_size=3, strides=1, padding='same', activation='relu')(net)
    net = Subpixel(filters=3, kernel_size=3, r=upscale_factor, padding='same')(net)

    outputs = Activation('relu')(net)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    model.save('model.h5')
    model.summary()

    history = model.fit_generator(train_gen, validation_data=val_gen, epochs=1, verbose=1, callbacks=[
        ModelCheckpoint(r"C:\labs\\project\\project\\model.h5",  # 풀 주소로 적어야 에러가 안 생김
                        monitor='val_loss', verbose=1, save_best_only=True)])

    return history


def sum1forline(filename):
    with open(filename) as f:
        return sum(1 for line in f)

# base_path = r"C:\labs\\project\\project\\celeba-dataset"
def img_resolution_test(base_path):
    img_base_path = os.path.join(base_path, "test_image")
    target_img_path = os.path.join(base_path, "processed_test_test")

    eval_list = np.loadtxt(os.path.join(base_path, "list_eval_partition_test_test.csv"),
                           dtype=str, delimiter=',', skiprows=1)

    img_sample = cv2.imread(os.path.join(img_base_path, eval_list[0][0]))

    h, w, _ = img_sample.shape
    print(h, w)

    # 정사각형 이미지로 crop
    crop_sample = img_sample[int((h-w)/2):int(-(h-w)/2), :]

    # 이미지 4배 축소 후 normalize
    resized_sample = pyramid_reduce(crop_sample, downscale=4, multichannel=True) # 컬러채널 허용

    pad = int((crop_sample.shape[0] - resized_sample.shape[0]) / 2)

    padded_sample = cv2.copyMakeBorder(resized_sample, top=pad + 1, bottom=pad,
                                       left=pad + 1, right=pad, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))

    print(crop_sample.shape, padded_sample.shape)

    # fig, ax = plt.subplots(1, 4, figsize=(12, 5))
    # ax = ax.ravel()
    # ax[0].imshow(img_sample)
    # ax[1].imshow(crop_sample)
    # ax[2].imshow(resized_sample)
    # ax[2].imshow(cv2.resize(resized_sample, dsize=(45, 45)))
    # ax[3].imshow(padded_sample)
    # plt.show()

    downscale = 4

    # 이미지 test 할 파일 생성
    for i, e in enumerate(eval_list):
        if i == sum1forline(os.path.join(base_path, "list_eval_partition_test_test.csv")):
            break
        filename, ext = os.path.splitext(e[0])
        img_path = os.path.join(img_base_path, e[0])
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        crop = img[int((h - w) / 2):int(-(h - w) / 2), :]
        crop = cv2.resize(crop, dsize=(176, 176))
        resized = pyramid_reduce(crop, downscale=downscale, multichannel=True) # 컬러 채널 허용
        norm = cv2.normalize(crop.astype(np.float64), None, 0, 1, cv2.NORM_MINMAX)

        if int(e[1]) == 2:  # Test
            np.save(os.path.join(target_img_path, "x_test", filename + ".npy"), resized)
            np.save(os.path.join(target_img_path, "y_test", filename + ".npy"), norm)

    return None


# base_path = r"C:\labs\\project\\project\\celeba-dataset\\processed_test"
def img_resolution_test2(base_path):
    x_test_list = sorted(glob.glob(os.path.join(base_path, 'x_test', '*.npy')))
    y_test_list = sorted(glob.glob(os.path.join(base_path, 'y_test', '*.npy')))
    print(len(x_test_list), len(y_test_list))
    # print(x_test_list[0])

    test_idx = -1

    # 저해상도 이미지(input)
    x1_test = np.load(x_test_list[test_idx])

    # 저해상도 이미지 확대시킨 이미지
    x1_test_resized = pyramid_expand(x1_test, 4, multichannel=True)  #색깔 채널 조건 추가

    # 정답 이미지
    y1_test = np.load(y_test_list[test_idx])

    # 모델 만들기
    upscale_factor = 4
    inputs = Input(shape=(44, 44, 3))

    net = Conv2D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu')(inputs)
    net = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')(net)
    net = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(net)
    net = Conv2D(filters=upscale_factor ** 2, kernel_size=3, strides=1, padding='same', activation='relu')(net)
    net = Subpixel(filters=3, kernel_size=3, r=upscale_factor, padding='same')(net)

    outputs = Activation('relu')(net)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    # model.save('model.h5')

    # 모델이 예측한 이미지(output)
    y_pred = model.predict(x1_test.reshape((1, 44, 44, 3)))
    print(x1_test.shape, y1_test.shape)

    x1_test = (x1_test * 255).astype(np.uint8)
    x1_test_resized = (x1_test_resized * 255).astype(np.uint8)
    y1_test = (y1_test * 255).astype(np.uint8)
    y_pred = np.clip(y_pred.reshape((176, 176, 3)), 0, 1)

    x1_test = cv2.cvtColor(x1_test, cv2.COLOR_BGR2RGB)
    x1_test_resized = cv2.cvtColor(x1_test_resized, cv2.COLOR_BGR2RGB)
    y1_test = cv2.cvtColor(y1_test, cv2.COLOR_BGR2RGB)
    y_pred = cv2.cvtColor(y_pred, cv2.COLOR_BGR2RGB)

    # fig, ax = plt.subplots(1,4,figsize=(15, 10))
    # ax = ax.ravel()
    #
    # ax[0].set_title('input')
    # ax[0].imshow(x1_test)
    #
    # ax[1].set_title('resized')
    # ax[1].imshow(x1_test_resized)
    #
    # ax[2].set_title('output')
    # ax[2].imshow(y_pred)
    #
    # ax[3].set_title('groundtruth')
    # ax[3].imshow(y1_test)
    #
    # plt.show()

    return None

def del_csv():
    df = pd.read_csv('../project/celeba-dataset/list_eval_partition_test_test.csv')
    df = df.drop([df.index[-1]])
    df.to_csv('../project/celeba-dataset/list_eval_partition_test_test.csv', index=False)


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

ALL = list(range(0, 68))


def test_image_df_recall(name_number):
    with open(f'../project/dataframe_test/{name_number}.pkl', mode='rb') as f:
        p = pickle.load(f)
        df2 = pd.DataFrame(p)
    return df2


# parameter 값으로 자신의 이미지 파일을 저장해 놓은 경로를 입력한다.
def facing_67_test(face_folder_path, rank):
    predictor_path = "../testing/shape-predict/shape_predictor_68_face_landmarks.dat"
    faces_folder_path = face_folder_path

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    # dict를 저장하는 리스트
    landmark_dict_list = []

    for f in glob.glob(os.path.join(faces_folder_path, f"{rank}*.jpg")):
        img = dlib.load_rgb_image(f)  # directory 에 있는 이미지 파일 하나씩 불러오기
        r, g, b = cv2.split(img)  # bgr을 rgb 로변경
        cvImg = cv2.merge([b, g, r])
        cvImg = cv2.bilateralFilter(cvImg, 9, 75, 75)

        dets = detector(cvImg, 1)  # face detecting function. 이 결과로 rectangle[] value return
        if len(dets) == 1:  # rectangle[] 에 값이 하나인 경우.
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
                    landmark_dict = dict()
                    for j in ALL:
                        landmark_dict[j] = landmark_list[j]
                    landmark_dict_list.append(landmark_dict)
                    with open(f'../testing/json_test/test_{rank}.json', "w") as json_file:
                        json_file.write(json.dumps(landmark_dict_list))
                        json_file.write('\n')
                else:
                    os.remove(f)
                    del_csv()

        else:
            os.remove(f)
            del_csv()

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
    print(t)
    print(f'당신과 가장 닮은 부자는 {data.index[np.argmin(clist)]}입니다.')
    if t == 0:
        return(f'자수성가형!\n뼈빠지게 고생하는 타입인 당신!\n언젠간 포브스 500에 당신의 이름이 실릴 날이...\n(ex. 제프 베조스)\n당신과 가장 닮은 부자는 {data.index[np.argmin(clist)]}입니다.')
    elif t == 1:
        return(f'금수저형!\n꽝! 다음 생애에... (ex. 이건희)\n당신과 가장 닮은 부자는 {data.index[np.argmin(clist)]}입니다.')
    elif t == 2:
        return(f'투자의귀재!\n보는 눈이 있는 당신!\n손대는 것마다 투자성공!\n(ex. 워렌 버핏)\n당신과 가장 닮은 부자는 {data.index[np.argmin(clist)]}입니다.')
    elif t == 3:
        return(f'또라이형!\n 모 아니면 도! 포브스와 정신병원의 갈림길에 서 있는 당신!\n(ex. 일론머스크)\n당신과 가장 닮은 부자는 {data.index[np.argmin(clist)]}입니다.')
    elif t == 4:
        return(f'자퇴형!\n 일단 자퇴를 해라. \n그러면 성공할 것이다.\n(ex. 주커버그, 빌게이츠, ...)\n당신과 가장 닮은 부자는 {data.index[np.argmin(clist)]}입니다.')
    elif t == 5:
        return(f'결혼형!\n 배우자 복이 있는 당신!\n 행복하세요! (ex. 맥킨지 베조스)\n당신과 가장 닮은 부자는 {data.index[np.argmin(clist)]}입니다.')
    elif t == 6:
        return(f'시인형!\n 당신은 부자의 관상은 아닙니다.. 하지만!\n17세기 최고의 시인이 될 수 있는 관상!\n돈보다 문학을 선택한 당신! 화이팅! (ex. 월터스콧)\n당신과 가장 닮은 부자는 {data.index[np.argmin(clist)]}입니다.')
    elif t == 7:
        return(f'UN특사형!\n 당신은 부자의 관상은 아닙니다.. 하지만!\n국제무대의 비둘기가 될 수 있는 관상!\nun특사에 도전해보는 것은 어떨까요?! (ex. Peter Thomson)\n당신과 가장 닮은 부자는 {data.index[np.argmin(clist)]}입니다.')
    elif t == 8:
        return(f'정치인형!\n 당신은 부자의 관상은 아닙니다.. 하지만!\n뒷주머니로 챙기기 나름!\n 정치로 세계 500대 부자에 도전해 보세요! \n(ex.Elaine Marshall)\n당신과 가장 닮은 부자는 {data.index[np.argmin(clist)]}입니다.')
    elif t == 9:
        return(f'professor type!\n 당신은 부자의 관상이 아닙니다..\n 하지만! 대학원 5년을 견디면...\n 교수가 될, 수도 있는 당신! 화이팅..\n (ex. Mary Malone)\n당신과 가장 닮은 부자는 {data.index[np.argmin(clist)]}입니다.')


def plot_radar_chart(name_number=1000,
                     color='r', percent=False):
    rich_mean = [0.257579, 0.162942, 0.162557, 0.272610, 0.202978,
                 0.265984, 0.290559, 0.268134, 0.381116, 0.107953]

    if name_number == 1000:  # isinstance(name_number, list)
        new_df = rich_mean
    else:
        new_df = test_image_df_recall(name_number)[
            ['eye_between', 'left_eye_width', 'right_eye_width', 'left_eyebrow', 'nose_width',
             'right_eyebrow', 'nose_height', 'jaw_height', 'mouth_width', 'philtrum']]
        new_df = new_df.values.tolist()[0]

    dif = []
    for x in range(10):
        dif.append(round((1 - abs(rich_mean[x] - new_df[x]) / rich_mean[x]) * 100, 2))
    print('dif :', dif)
    print(np.mean(dif))

    # Font
    fm.get_fontconfig_fonts()
    # font_location = '/usr/share/fonts/truetype/nanum/NanumGothicOTF.ttf'
    font_location = 'C:/windows/fonts/gaesung.ttf'  # For Windows
    font_name = fm.FontProperties(fname=font_location).get_name()
    plt.rc('font', family=font_name, size=10)


    cat = ['eye_between', 'left_eye_width', 'right_eye_width', 'left_eyebrow', 'nose_width',
           'right_eyebrow', 'nose_height', 'jaw_height', 'mouth_width', 'philtrum']
    values = new_df

    N = len(cat)

    x_as = [n / float(N) * 2 * pi for n in range(N)]

    # Because our chart will be circular we need to append a copy of the first
    # value of each list at the end of each list with data
    values += values[:1]
    x_as += x_as[:1]
    # print(x_as)

    # Set color of axes
    plt.rc('axes', linewidth=0.5, edgecolor="#888888")


    # Create polar plot
    ax = plt.subplot(121, polar=True)

    # Set clockwise rotation. That is:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Set position of y-labels
    ax.set_rlabel_position(0)

    # Set color and linestyle of grid
    ax.xaxis.grid(True, color="#888888", linestyle='solid', linewidth=0.5)
    ax.yaxis.grid(True, color="#888888", linestyle='solid', linewidth=0.5)

    # Set number of radial axes and remove labels
    plt.xticks(x_as[:-1], [])

    # Set yticks
    plt.yticks([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5], ["0.05", "0.1", "0.15", "0.2", "0.25", "0.3", "0.4"])

    # Plot data
    ax.plot(x_as, values, linewidth=2, linestyle='solid', zorder=3, marker='o', markersize=5)

    # Fill area
    ax.fill(x_as, values, color, alpha=0.3)

    # Set axes limits
    plt.ylim(0, 0.5)

    if percent==True:
        ax.set_title(f'{round(np.mean(dif), 4)} %', size=18,
                     color='red', rotation='horizontal')

    # Draw ytick labels to make sure they fit properly
    for i in range(N):
           angle_rad = i / float(N) * 2 * pi
           if angle_rad == 0:
               ha, distance_ax = "center", 0.55
           elif 0 < angle_rad < pi:
               ha, distance_ax = "left", 0.5
           elif angle_rad == pi:
               ha, distance_ax = "center", 0.55
           else:
               ha, distance_ax = "right", 0.5

           ax.text(angle_rad, distance_ax, cat[i], size=12, horizontalalignment=ha, verticalalignment="center",
                   fontstyle='oblique')
    # if score == True:
    #     for i in range(N):
    #         angle_rad = i / float(N) * 2 * pi
    #         if angle_rad == 0:
    #             ha, distance_ax = "center", 0.6
    #         elif 0 < angle_rad < pi:
    #             ha, distance_ax = "left", 0.6
    #         elif angle_rad == pi:
    #             ha, distance_ax = "center", 0.6
    #         else:
    #             ha, distance_ax = "right", 0.6
    #
    #
    #         ax.text(angle_rad, distance_ax, f'{dif[i]} %',color='blue', size=10, horizontalalignment=ha, verticalalignment="center")

    # Show polar plot
    # fig.patch.set_visible(False)

    ax2 = plt.subplot(122)
    ax2.axis('off')
    ax2.axis('tight')
    clust_data = ['eye_between', 'left_eye_width', 'right_eye_width', 'left_eyebrow', 'nose_width',
                  'right_eyebrow', 'nose_height', 'jaw_height', 'mouth_width', 'philtrum']
    percent_data = dif

    collabel_1 = ['location']
    df = pd.DataFrame(clust_data, columns=collabel_1)
    df['percent(%)'] = percent_data
    print(df)
    ax2.table(cellText=df.values, colLabels=df.columns, loc='center',
              colLoc='center', cellLoc='center')

    if name_number == 1000:  # isinstance(name_number, list)
        pass
    else:
        name_number = name_number
        ax2.set_title(f'{return_type(name_number)}', size=18,  rotation='horizontal')


def final_train():
    img_resolution_train(r"C:\labs\\project\\project\\celeba-dataset")
    img_resolution_train2(r"C:\labs\\project\\project\\celeba-dataset\\processed")

    return None

def last_image_save():
    files = sorted(glob.glob(os.path.join("../project/celeba-dataset/test_image/", '*.jpg')), key=os.path.getctime,
                   reverse=True)
    number = np.int(files[0][37:41])

    with open(os.path.join("../project/celeba-dataset/", 'list_eval_partition_test_test.csv'), newline='',
              mode='a') as file:
        wr = csv.writer(file)
        files = sorted(glob.glob(os.path.join("../project/celeba-dataset/test_image/", f'{number}.jpg')),
                       key=os.path.getctime,
                       reverse=True)
        wr.writerow([files[0][37:], 2])

    src = cv2.imread(str(f"../project/celeba-dataset/test_image/{number}.jpg"),
                     cv2.IMREAD_COLOR)
    print(f"../project/celeba-dataset/test_image/{number}.jpg")
    resizing = cv2.resize(src, dsize=(178, 218), interpolation=cv2.INTER_CUBIC)
    gtorgb = cv2.cvtColor(resizing, cv2.COLOR_BGR2RGB)
    blue_mosters = imageio.imwrite(f"../project/celeba-dataset/test_image/{number}.jpg",
                                   im=gtorgb, pilmode='CMYK', as_gray=False)

    img_resolution_test(r"C:\labs\\project\\project\\celeba-dataset")
    img_resolution_test2(r"C:\labs\\project\\project\\celeba-dataset\\processed_test_test")

    facing_67_test("../project/celeba-dataset/test_image", number)

    df = pd.DataFrame(columns=['right_eye_width', 'right_eye_height', 'right_eye_shape', 'right_eyebrow',
                               'left_eye_width', 'left_eye_height', 'left_eye_shape', 'left_eyebrow',
                               'eye_between', 'nose_width', 'nose_height', 'philtrum',
                               'mouth_width', 'mouth_height', 'mouth_shape', 'jaw_height'])
    for j in glob.glob(os.path.join('../testing/json_test/', f'test_{number}.json')):
        with open(j, mode='r') as file:
            lst = json.load(file)
            for k in range(len(lst)):
                dc = json_to_dict(j)[k]
                pt2 = FaceDistanceRatio(dc)
                data = pt2.dict_to_series()
                df.loc[k] = data
            print(df)
            with open(f'../project/dataframe_test/{number}.pkl', mode='wb') as pic:
                pickle.dump(df, pic)
    return None



if __name__ == '__main__':
    # final_train()
    # img_resolution_train(r"C:\labs\\project\\project\\celeba-dataset")
    # img_resolution_train2(r"C:\labs\\project\\project\\celeba-dataset\\processed")

    last_image_save()

    # 실행전 확인할 것
    # '../project/celeba-dataset/test_image' 폴더 생성 사진파일 1개 넣어두기
    # '../project/celeba-dataset/list_eval_partition_test_test.csv' 파일 생성 아무 사진이나 새로만든 test_image폴더에 넣어두고 1줄 작성
    # '../project/celeba-dataset/proceseed_test_test' 폴더 생성 이때 폴더 안에 'x_test', 'y_test'폴더 도 추가로 생성
    # '../testing/json_test' 폴더 생성
    # '../project/dataframe_test' 폴더 생성

    # files = sorted(glob.glob(os.path.join("../project/celeba-dataset/test_image/", '*.jpg')), key=os.path.getctime,
    #                reverse=True)
    # number = np.int(files[0][37:41])
    # with open(os.path.join("../project/celeba-dataset/", 'list_eval_partition_test_test.csv'), newline='',
    #           mode='a') as file:
    #     wr = csv.writer(file)
    #     files = sorted(glob.glob(os.path.join("../project/celeba-dataset/test_image/", f'{number}.jpg')), key=os.path.getctime,
    #                    reverse=True)
    #     wr.writerow([files[0][37:], 2])
    #
    # src = cv2.imread(str(f"../project/celeba-dataset/test_image/{number}.jpg"),
    #                  cv2.IMREAD_COLOR)
    # print(f"../project/celeba-dataset/test_image/{number}.jpg")
    # resizing = cv2.resize(src, dsize=(178, 218), interpolation=cv2.INTER_CUBIC)
    # gtorgb = cv2.cvtColor(resizing, cv2.COLOR_BGR2RGB)
    # blue_mosters = imageio.imwrite(f"../project/celeba-dataset/test_image/{number}.jpg",
    #                                im=gtorgb, pilmode='CMYK', as_gray=False)
    #
    # img_resolution_test(r"C:\labs\\project\\project\\celeba-dataset")
    # img_resolution_test2(r"C:\labs\\project\\project\\celeba-dataset\\processed_test_test")
    #
    # facing_67_test("../project/celeba-dataset/test_image", number)
    #
    #
    #
    # df = pd.DataFrame(columns=['right_eye_width', 'right_eye_height', 'right_eye_shape', 'right_eyebrow',
    #                            'left_eye_width', 'left_eye_height', 'left_eye_shape', 'left_eyebrow',
    #                            'eye_between', 'nose_width', 'nose_height', 'philtrum',
    #                            'mouth_width', 'mouth_height', 'mouth_shape', 'jaw_height'])
    # for j in glob.glob(os.path.join('../testing/json_test/', f'test_{number}.json')):
    #     with open(j, mode='r') as file:
    #         lst = json.load(file)
    #         for k in range(len(lst)):
    #             dc = json_to_dict(j)[k]
    #             pt2 = FaceDistanceRatio(dc)
    #             data = pt2.dict_to_series()
    #             df.loc[k] = data
    #         print(df)
    #         with open(f'../project/dataframe_test/{number}.pkl', mode='wb') as pic:
    #             pickle.dump(df, pic)

    plot_radar_chart()  # 부자들의 평균을 그리는 함수 모두 기본값
    plot_radar_chart(name_number=1010, color='yellow', percent=True)

    # plt.tight_layout()
    plt.show()

    # 0 자수성가형, 1 금수저형, 2 투자의귀재, 3 또라이형, 4 자퇴형
    # 5 결혼형, 6 시인형, 7 UN특사형, 8 정치인형, 9 professor type
    # type = [0, 4, 4, 0, 4, 1, 0, 0, 0, 4, 1, 0, 0, 1, 1, 1, 5, 1, 5, 3,
    #         0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 4, 0, 1, 0, 0, 1, 1, 5, 0, 0,
    #         1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 5, 1, 0, 1, 1, 0, 1, 0, 0, 8,
    #         0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 2, 1, 2, 0, 1, 0, 1, 2,
    #         2, 0, 0, 0, 0, 0, 1, 1, 0, 1, 2, 0, 1, 1, 1, 1, 4, 2, 0, 0,
    #         0, 0, 2, 0, 5, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
    #         4, 0, 1, 2, 2, 1, 6, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    #         0, 0, 0, 1, 1, 4, 1, 0, 0, 4, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0,
    #         2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0,
    #         0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 5, 0, 0, 1,
    #         0, 0, 1, 0, 1, 0, 4, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    #         0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1, 1, 2, 1, 2, 1,
    #         0, 0, 1, 1, 2, 0, 0, 1, 0, 1, 0, 2, 2, 4, 0, 5, 0, 0, 0, 0,
    #         0, 0, 0, 2, 0, 0, 1, 0, 5, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1,
    #         1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
    #         0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 2, 1, 1, 1, 2, 0, 1, 2,
    #         0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 2, 1, 0, 7, 1, 1, 0, 1,
    #         1, 1, 1, 0, 0, 1, 1, 1, 0, 2, 2, 0, 0, 1, 1, 1, 0, 1, 0, 1,
    #         0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 3, 1, 0, 3,
    #         1, 1, 1, 0, 1, 1, 0, 0, 0, 3, 0, 0, 1, 1, 1, 1, 6, 0, 0, 1,
    #         1, 0, 1, 1, 0, 0, 1, 2, 0, 1, 0, 0, 0, 0, 0, 1, 9, 1, 2, 2,
    #         1, 1, 1, 0, 5, 0, 0, 4, 5, 0, 0, 1, 4, 3, 0, 4, 1, 1, 0, 0,
    #         1, 3, 0, 0, 2, 2, 2, 0, 1, 1, 0, 4, 0, 1, 1, 1, 1, 1, 0, 1,
    #         1, 3, 5, 0, 0, 0, 1, 1, 1, 0, 1, 5, 1, 0, 0, 6, 1, 4, 0, 2,
    #         0, 1, 0, 0, 1, 1, 3, 0, 5, 1, 0, 0, 1, 1, 1, 1, 0, 1]

    # 0 자수성가형, 1 금수저형, 2 투자의귀재, 3 또라이형, 4 자퇴형
    # 5 결혼형, 6 시인형, 7 UN특사형, 8 정치인형, 9 professor type