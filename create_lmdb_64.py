import matplotlib.pyplot as plt
import csv
import numpy as np
import os
import glob
import random

import cv2

import caffe
from caffe.proto import caffe_pb2
import lmdb

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img

def make_datum(img, label):
    #image is numpy.ndarray format. BGR instead of RGB
    return caffe_pb2.Datum(
        channels=3,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        label=label,
        data=np.rollaxis(img, 2).tostring())

# function for reading the images
# arguments: path to the traffic sign data, for example './GTSRB/Training'
# returns: list of images, list of corresponding labels
def readTrafficSigns(rootpath):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    train_data = [] # images
    # loop over all 42 classes
    for c in range(0,43):
        prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        gtReader.next() # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            train_data.append([prefix + row[0], row[7]]) # the 1th column is the filename
        gtFile.close()
    return train_data

train_lmdb = 'input/train_lmdb_64'
validation_lmdb = 'input/validation_lmdb_64'


train_data = readTrafficSigns('GTSRB/Training')

#Shuffle train_data
random.shuffle(train_data)

print 'Creating train_lmdb_64'

in_db = lmdb.open(train_lmdb, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    for in_idx, img_path_label in enumerate(train_data):
        if in_idx %  6 == 0:
            continue
        img = cv2.imread(img_path_label[0], cv2.IMREAD_COLOR)
        img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
        label = int(img_path_label[1])
        datum = make_datum(img, label)
        in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
        print '{:0>5d}'.format(in_idx) + ':' + img_path_label[0] + "--" + img_path_label[1]
in_db.close()


print '\nCreating validation_lmdb_64'

in_db = lmdb.open(validation_lmdb, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    for in_idx, img_path_label in enumerate(train_data):
        if in_idx % 6 != 0:
            continue
        img = cv2.imread(img_path_label[0], cv2.IMREAD_COLOR)
        img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
        label = int(img_path_label[1])
        datum = make_datum(img, label)
        in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
        print '{:0>5d}'.format(in_idx) + ':' + img_path_label[0] + "--" + img_path_label[1]
in_db.close()

print '\nFinished processing all images'
