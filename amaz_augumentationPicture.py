import urllib.request
import tarfile
from os import system
import os
import sys
import cv2
import six
import pickle
from tqdm import tqdm
import numpy as np
import random

import amaz_sampling
sampling = amaz_sampling.Sampling()

class AugumentationPicture(object):

    def __init__(self):
        self.name = "augpic"

    @staticmethod
    def convert_to_imgAry(X):
        """
        * about:
        convert to normal image transpose
        """
        return np.transpose(X,(1,2,0))

    @staticmethod
    def convert_to_chainerVariable(X):
        """
        * about:
        convert image to chainer transpose
        """
        return np.transpose(X,(2,0,1))

    @staticmethod
    def flip_horizontal(X,probability):
        """
        * about:
        flip image depending on probability
        """
        seed = 100
        thold = probability * seed
        radval = random.randint(0,seed)
        if radval < thold:
            return cv2.flip(X,1)
        else:
            return X

    @staticmethod
    def resize(X,size,interpolation="LINER"):
        """
        * about:
        resize image to ideal size
        * detail:
        interpolation : (LINEAR or CUBIC)
        """
        if interpolation == "LINER":
            interp_setting = cv2.INTER_LINEAR
        else:
            interp_setting = cv2.INTER_CUBIC

        resimg = cv2.resize(X,size,interpolation=interp_setting)
        return resimg

    @staticmethod
    def crop_random(X,size):
        """
        * about:
        crop image to ideal size
        """
        y, x, channel = X.shape

        #calculate keypoints to crop
        sizeX,sizeY = size
        keypoint_y = sampling.pick_random_permutation(1, y - sizeY + 1)[0]
        keypoint_x = sampling.pick_random_permutation(1, x - sizeX + 1)[0]

        x, y, w, h = (keypoint_x, keypoint_y, sizeX, sizeY)
        return X[y:y+h, x:x+w]

    @staticmethod
    def normalize(X,value=0.):
        """
        * about:
        normalize image
        * equation
        (x - mean) / sqrt(variance + value)
        """
        var = np.var(X)
        std = np.sqrt(var + value)
        mean = np.mean(X)
        return (X - mean) / std

    @staticmethod
    def normalize_locally(X,value=0.):
        number_of_picture, length = len(X), len(X[0])
        # calculate variance
        # add value here
        var = np.var(X, axis=1) + value
        # increase dimension: 1 to 2
        var = np.reshape(var, (len(var), 1))
        var = np.repeat(var, length, axis=1)
        std = np.sqrt(var)
        # calculate mean
        mean = np.mean(X, axis=1)
        # increase dimension: 1 to 2
        mean = np.reshape(mean, (len(mean), 1))
        mean = np.repeat(mean, length, axis=1)
        return np.subtract(X, mean) / std
