from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import cv2
import imageio
import glob
import os
import dlib
from sklearn.decomposition import PCA
import PIL.Image as pilimg
import numpy as np


def drawing_master(faces, labels, ncol=5):
    """

    :param faces:
    :param labels:
    :param ncol:
    :return:
    """
    n_faces = len(faces)
    nrow = (n_faces - 1) // ncol + 1
    for i, (face, label) in enumerate(zip(faces, labels)):
        plt.subplot(nrow, ncol, i + 1)
        plt.imshow(face.reshape((64, 64)), cmap=plt.cm.gray)
        plt.title(label)
        plt.axis('off')
        plt.show()

path = '../project/celeba-dataset/crop_img'
def gaussian_mixture(path, rich_number):
    aa = []
    for f in glob.glob(os.path.join(path, f'{rich_number}*.jpg')):
        img = pilimg.open(f)
        img = np.array(img)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, dsize=(64, 64))
        nx, ny = img.shape
        print(img.shape)
        test_num = img.reshape((nx * ny,))
        print(test_num.shape)
        aa.append(test_num)

    pca = PCA(n_components=0.95, random_state=55)  # 모델 생성
    test_num_pca = pca.fit_transform(aa)
    gm = GaussianMixture(n_components=len(aa), random_state=55)
    gm.fit(test_num_pca)
    gm_faces, gm_labels = gm.sample(n_samples=1)

    gm_faces = pca.inverse_transform(gm_faces)
    drawing_master(gm_faces, gm_labels)


if __name__ == '__main__':
    gaussian_mixture('../project/celeba-dataset/crop_img', 103)