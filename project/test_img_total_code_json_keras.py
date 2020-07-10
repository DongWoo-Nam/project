import csv
import pickle

import imageio
import os
import dlib
import glob
import cv2
import numpy as np
import json
import math
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import pyramid_reduce
import warnings

from project.class01_Face_Distance_Ratio import FaceDistanceRatio
from project.json_to_dict_function import json_to_dict

warnings.filterwarnings(action='ignore')
from keras.layers import Conv2D, Input, Activation
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from skimage.transform import pyramid_expand
from project.keras_01_Subpixel import Subpixel
from project.keras_02_Datagenerator import DataGenerator





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



if __name__ == '__main__':
    # img_resolution_train(r"C:\labs\\project\\project\\celeba-dataset")
    # img_resolution_train2(r"C:\labs\\project\\project\\celeba-dataset\\processed")

    # 실행전 확인할 것
    # '../project/celeba-dataset/test_image' 폴더 생성 사진파일 1개 넣어두기
    # '../project/celeba-dataset/list_eval_partition_test_test.csv' 파일 생성 아무 사진이나 새로만든 test_image폴더에 넣어두고 1줄 작성
    # '../project/celeba-dataset/proceseed_test_test' 폴더 생성 이때 폴더 안에 'x_test', 'y_test'폴더 도 추가로 생성
    # '../testing/json_test' 폴더 생성
    # '../project/dataframe_test' 폴더 생성

    rank = 1001
    for i in range(10):  # 폴더에 있는 사진의 개수
        with open(os.path.join("../project/celeba-dataset/", 'list_eval_partition_test_test.csv'), newline='',
                  mode='a') as file:
            wr = csv.writer(file)
            files = sorted(glob.glob(os.path.join("../project/celeba-dataset/test_image/", f'{rank}.jpg')), key=os.path.getctime,
                           reverse=True)
            wr.writerow([files[0][37:], 2])

        src = cv2.imread(str(f"../project/celeba-dataset/test_image/{rank}.jpg"),
                         cv2.IMREAD_COLOR)
        print(f"../project/celeba-dataset/test_image/{rank}.jpg")
        resizing = cv2.resize(src, dsize=(178, 218), interpolation=cv2.INTER_CUBIC)
        gtorgb = cv2.cvtColor(resizing, cv2.COLOR_BGR2RGB)
        blue_mosters = imageio.imwrite(f"../project/celeba-dataset/test_image/{rank}.jpg",
                                       im=gtorgb, pilmode='CMYK', as_gray=False)

        img_resolution_test(r"C:\labs\\project\\project\\celeba-dataset")
        img_resolution_test2(r"C:\labs\\project\\project\\celeba-dataset\\processed_test_test")

        facing_67_test("../project/celeba-dataset/test_image", rank)
        rank += 1


    pt1 = FaceDistanceRatio(json_to_dict('../testing/json_test/test_1001.json')[0])
    print(pt1.dict_to_df('test'))  # index는 항상 []로 들어와야함 이름을 뽑을 때에도
    print()
    df = pd.DataFrame(columns=['right_eye_width', 'right_eye_height', 'right_eye_shape', 'right_eyebrow',
                               'left_eye_width', 'left_eye_height', 'left_eye_shape', 'left_eyebrow',
                               'eye_between', 'nose_width', 'nose_height', 'philtrum',
                               'mouth_width', 'mouth_height', 'mouth_shape', 'jaw_height'])
    for i in range(11):    # 폴더에 있는 사진의 개수 + 1
        for j in glob.glob(os.path.join('../testing/json_test/', f'test_{1000 + i}.json')):
            with open(j, mode='r') as file:
                lst = json.load(file)
                for k in range(len(lst)):
                    dc = json_to_dict(j)[k]
                    pt2 = FaceDistanceRatio(dc)
                    data = pt2.dict_to_series()
                    df.loc[k] = data
                print(df)
                with open(f'../project/dataframe_test/{1000 + i}.pkl', mode='wb') as pic:
                    pickle.dump(df, pic)

    print(test_image_df_recall(1001))
    # [8, 4, 0, 7, 9, 3, 10, 15, 12, 11]
    new_df = test_image_df_recall(1001)[['eye_between', 'left_eye_width', 'right_eye_width', 'left_eyebrow', 'nose_width',
                                         'right_eyebrow', 'nose_height', 'jaw_height', 'mouth_width', 'philtrum']]
    print(new_df)
    print(new_df.values)
    print(new_df.values.tolist()[0])


