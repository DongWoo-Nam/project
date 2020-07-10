import warnings
warnings.filterwarnings(action='ignore')

import os, glob
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Conv2D, Input, Activation
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from skimage.transform import pyramid_expand
from project.keras_01_Subpixel import Subpixel
from project.keras_02_Datagenerator import DataGenerator


base_path = r"C:\labs\\project\\project\\celeba-dataset\\processed"

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

    history = model.fit_generator(train_gen, validation_data=val_gen, epochs=5, verbose=1, callbacks=[
        ModelCheckpoint(r"C:\labs\\project\\project\\model.h5",  # 풀 주소로 적어야 에러가 안 생김
                        monitor='val_loss', verbose=1, save_best_only=True)])

    return history