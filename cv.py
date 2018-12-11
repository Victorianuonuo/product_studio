from __future__ import print_function

import os
import numpy as np

from skimage.io import imsave, imread
from skimage.transform import resize
from pathlib import Path
from tqdm import tqdm
from skimage import io
from scipy.io import loadmat
import cv2

from data_keras_512 import load_train_data, load_test_data
from main_keras_512 import get_unet
import matplotlib.pyplot as plt

import pydensecrf.densecrf as dcrf

def dense_crf(img, output_probs):
    h = output_probs.shape[0]
    w = output_probs.shape[1]

    output_probs = np.expand_dims(output_probs, 0)
    output_probs = np.append(1 - output_probs, output_probs, axis=0)

    d = dcrf.DenseCRF2D(w, h, 2)
    U = -np.log(output_probs)
    U = U.reshape((2, -1))
    U = np.ascontiguousarray(U)
    img = np.ascontiguousarray(img)

    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=20, compat=3)
    d.addPairwiseBilateral(sxy=30, srgb=20, rgbim=img, compat=10)

    Q = d.inference(5)
    Q = np.argmax(np.array(Q), axis=0).reshape((h, w))

    return Q

""" 
Comments:

'./checkpoints/weights_11_1.h5': 96*96 image size.
from data_keras import load_train_data, load_test_data
from main_keras import get_unet

'./checkpoints/weights_11_2.h5': 512*512 image size: 【best】
from data_keras_512 import load_train_data, load_test_data
from main_keras_512 import get_unet
img[mask > 0.6] = (0, 0, 255)

'./checkpoints/weights_11_2_epoch_32.h5': 512*512 image size:
from data_keras_512 import load_train_data, load_test_data
from main_keras_512 import get_unet
img[mask > 0.0000001] = (0, 0, 255) # current best: 0.00001

'./checkpoints/weights_11_3_epoch_47.h5': 512*512 image size:
img[mask > 0.6] = (0, 0, 255)
from main_keras_65_layers import get_unet
"""

image_rows = 512
image_cols = 512

ANNOTATION_PATH = './clothing-co-parsing/annotations/pixel-level/'
MODEL_PATH = './checkpoints/weights_11_2.h5'
N_EPOCH = 250
N_train_start = 0 
N_train_end = 1004
N_test_start = 1005
N_test_end = 1015

model = get_unet()
model.load_weights(MODEL_PATH)

imgs_train, imgs_mask_train = load_train_data()
imgs_train = imgs_train[..., np.newaxis]
imgs_mask_train = imgs_mask_train[..., np.newaxis]
imgs_train = imgs_train.astype('float32')
mean = np.mean(imgs_train)  # mean for data centering
std = np.std(imgs_train)  # std for data normalization

def webcam():
    # face_cascade = cv2.CascadeClassifier('frontalface_default.xml')
    # specs_ori = cv2.imread('glass/glass.png', -1)
    # cigar_ori = cv2.imread('mouth/cigar.png',-1)
    cv2.namedWindow("preview")
    cam = cv2.VideoCapture(0) #webcame video
    # cap = cv2.VideoCapture('jj.mp4') #any Video file also
    cam.set(cv2.CAP_PROP_FPS, 30)
    ret, img = cam.read()
    
    while True:
        if img is not None:            
            # print(img.shape)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized =  cv2.resize(gray, dsize=(image_rows, image_cols), interpolation=cv2.INTER_CUBIC)
            input_img = np.ndarray((1, image_rows, image_cols), dtype=np.uint8)
            input_img[0] = resized
            input_img = input_img[..., np.newaxis]
            input_img = input_img.astype('float32')
            input_img -= mean
            input_img /= std

            mask = model.predict(input_img)
            mask = mask[0]
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            # print(mask.shape)
            
            ###
            # mask = dense_crf(img, mask)
            # mask = mask > 0.1
            ###

            img[mask > 0.6] = (0, 0, 255)
            # img[mask > 0.6] = (0, 0, 255)
            # img[mask <= 0] = (255, 255, 255)

            cv2.imshow('cloth segment', img)

            # out = cv2.imwrite('capture.jpg', img)
            # break
            # cv2.imshow('cloth segment', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            out = cv2.imwrite('capture.jpg', img)
            break
        ret, img = cam.read()
    
    cam.release()
    
    cv2.destroyAllWindows()

def predict():
    file_path = './clothing-co-parsing/photos/'
    imgs = []
    
    file_path = Path(file_path)
    files = sorted(list(Path(file_path).iterdir()))
    i = 0
    for file in tqdm(files[N_test_start:N_test_end]): # only process N_train images.
        img = cv2.imread(str(file)) # into grayscale.
        imgs.append(img)
        # imgs[i] = cv2.resize(img, dsize=(image_rows, image_cols), interpolation=cv2.INTER_CUBIC)
    # np.save('imgs_test.npy', imgs)
    # print('Saving to .npy files done.')
    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    imgs_test = load_test_data()
    imgs_test_copy = imgs_test


    imgs_test = imgs_test[..., np.newaxis]
    # imgs_test = preprocess(imgs_test)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model = get_unet()
    model.load_weights(MODEL_PATH)

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_mask_test = model.predict(imgs_test, verbose=1)
    np.save('imgs_mask_test.npy', imgs_mask_test)

    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)
    pred_dir = 'preds'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for i, mask in enumerate(imgs_mask_test):
        # mask = (mask[:, :, 0] * 255.).astype(np.uint8)
        mask = (mask * 255.).astype(np.uint8)
        mask = cv2.resize(mask, (imgs[i].shape[1], imgs[i].shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = mask[..., np.newaxis]
        output = ((0.2 * imgs[i]) + (0.8 * mask)).astype("uint8")
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(imgs[i], interpolation='none')
        plt.subplot(1,2,2)
        plt.imshow(output, interpolation='none')
        plt.savefig(os.path.join(pred_dir, str(i) + '_pred.png'))

if __name__ == '__main__':
    webcam()
    # predict()