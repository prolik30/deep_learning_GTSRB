import os
import glob
import cv2
import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2
import csv

caffe.set_mode_gpu() 

#Size of images
IMAGE_WIDTH = 250
IMAGE_HEIGHT = 250

'''
Image processing helper function
'''

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img


'''
Reading mean image, caffe model and its weights 
'''
#Read mean image
mean_blob = caffe_pb2.BlobProto()
with open('/home/user1/GTSRB/input/mean.binaryproto') as f:
    mean_blob.ParseFromString(f.read())
mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
    (mean_blob.channels, mean_blob.height, mean_blob.width))


#Read model architecture and trained model's weights
net = caffe.Net('/home/user1/GTSRB/caffe_models/caffenet_deploy.prototxt',
                '/home/user1/GTSRB/caffe_models/caffenet_model_iter_20000.caffemodel',
                caffe.TEST)

#Define image transformers
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', mean_array)
transformer.set_transpose('data', (2,0,1))

def readTrafficSigns(rootpath):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images '''
    images = [] # images
    prefix = rootpath + '/'
    gtFile = open(prefix + 'GT-online_test.test.csv') # annotations file
    gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
    gtReader.next() # skip header
    # loop over all images in current annotations file
    for row in gtReader:
        images.append(prefix + row[0]) # the 1th column is the filename
    gtFile.close()
    return images

'''
Making predicitions
'''
#Reading image paths
test_img_paths = readTrafficSigns('GTSRB/Online-Test/Images')

#Making predictions
test_img_names = []
preds = []
for img_path in test_img_paths:
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
    
    net.blobs['data'].data[...] = transformer.preprocess('data', img)
    out = net.forward()
    pred_probas = out['prob']

    test_img_names = test_img_names + [img_path.split('/')[-1]]
    preds = preds + [pred_probas.argmax()]

    print img_path
    print pred_probas.argmax()
    print '-------'

'''
Making submission file
'''
with open("/home/user1/GTSRB/caffe_models/submission_caffenet_model.csv","w") as f:
    f.write("id,label\n")
    for i in range(len(test_img_names)):
        f.write(str(test_img_names[i])+"; "+str(preds[i])+"\n")
f.close()
