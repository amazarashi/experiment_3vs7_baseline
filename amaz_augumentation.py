import urllib.request
import tarfile
from os import system
import os
import sys
import six
import pickle
from tqdm import tqdm
import numpy as np
import amaz_augumentationPicture

augmuent_pic = amaz_augumentationPicture.AugumentationPicture()

class Augumentation(object):

    def __init__(self):
        self.name = "aug"

    def Z_score(self,data):
        """
        - detail
          mean : 0 , Variable : 1
        """
        l = data
        l2 = l - l.mean()
        l3 = l2 /l2.std()
        return l3

    def random_brightnetss(data,max_delta=50):
        """
        - detail
          add random val in particular-range to data-img
        """
        delta = np.random.uniform(-max_delta,max_delta)
        resimg = data + delta
        return resimg
