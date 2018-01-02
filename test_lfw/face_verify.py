import _init_paths
import caffe
import cv2
import numpy as np

import numpy as np
import os
import skimage
import sys
import caffe
import sklearn.metrics.pairwise as pw
import math
from fr_wuqianliang import *
#  sys.path.insert(0, '/Downloads/caffe-master/python');
#  load Caffe model

caffe.set_mode_gpu();
global net;
net = caffe.Classifier('deepID1_deploy.prototxt', 'snapshot_iter_6099296.caffemodel');

def compare_pic(feature1, feature2):
    predicts = pw.cosine_similarity(feature1, feature2);
    return predicts;

def get_feature(path, mean_blob):
    global net;
    X = read_image(path, mean_blob);
    #print X
    # test_num = np.shape(X)[0];
    # print test_num;
    out = net.forward_all(data = X);
    feature = np.float64(out['deepid_1']);
    feature = np.reshape(feature, (1, 256));
    return feature;

def get_feature_crop_align_cam(image_data,mean_blob):

    X = np.empty((1,3,96,96));
    crop_image = crop_align_image(image_data)
    if crop_image is not None:
        image = skimage.transform.resize(crop_image, (96, 96))*255;
        mean_blob.shape = (-1, 1);
        mean = np.sum(mean_blob) / len(mean_blob);
        X[0,0,:,:] = (image[:,:,0] - mean).reshape(1,96,96);
        X[0,1,:,:] = (image[:,:,1] - mean).reshape(1,96,96);
        X[0,2,:,:] = (image[:,:,2] - mean).reshape(1,96,96);
        out = net.forward_all(data = X);
        feature = np.float64(out['deepid_1']);
        feature = np.reshape(feature, (1, 256));
        return feature;
    return None

def get_feature_crop_align(filepath,mean_blob):

    X = np.empty((1,3,96,96));
    filename = filepath.split('\n');
    filename = filename[0];
    img_ori = cv2.imread(filename)
    crop_image = crop_align_image(img_ori)
    if crop_image is not None:
        image = skimage.transform.resize(crop_image, (96, 96))*255;
        mean_blob.shape = (-1, 1);
        mean = np.sum(mean_blob) / len(mean_blob);
        X[0,0,:,:] = (image[:,:,0] - mean).reshape(1,96,96);
        X[0,1,:,:] = (image[:,:,1] - mean).reshape(1,96,96);
        X[0,2,:,:] = (image[:,:,2] - mean).reshape(1,96,96);
        out = net.forward_all(data = X);
        feature = np.float64(out['deepid_1']);
        feature = np.reshape(feature, (1, 256));
        return feature;
    return None

def read_image(filepath, mean_blob):
    # averageImg = [129.1863, 104.7624, 93.5940];
    X = np.empty((1,3,96,96));
    filename = filepath.split('\n');
    filename = filename[0];
    im = skimage.io.imread(filename, as_grey=False);
    image = skimage.transform.resize(im, (96, 96))*255;
    mean_blob.shape = (-1, 1); 
    mean = np.sum(mean_blob) / len(mean_blob);
    X[0,0,:,:] = (image[:,:,0] - mean).reshape(1,96,96);
    X[0,1,:,:] = (image[:,:,1] - mean).reshape(1,96,96);
    X[0,2,:,:] = (image[:,:,2] - mean).reshape(1,96,96);

    return X;


if __name__ == '__main__':
    thershold = 0.85;
    TEST_SUM = 6000;
    DATA_BASE = "../../../../pic";
    MEAN_FILE = 'DeepID1_mean.proto';
    
    mean_blob = caffe.proto.caffe_pb2.BlobProto();
    mean_blob.ParseFromString(open(MEAN_FILE, 'rb').read());
    mean_npy = caffe.io.blobproto_to_array(mean_blob);
    
    print mean_npy.shape
     
    # read pic and name
    feature_arr = {}
    for img_name in os.listdir(DATA_BASE):
        feat = get_feature_crop_align(os.path.join(DATA_BASE,img_name),mean_npy)
        if feat is not None:
            feature_arr[img_name]=feat

    cap = cv2.VideoCapture(0)
    start = time()
    while True:

        ret, img = cap.read()

        feature_cam_face_crop_align = get_feature_crop_align_cam(img,mean_npy)
        if feature_cam_face_crop_align is not None:
            for k in feature_arr:
                result = compare_pic(feature_cam_face_crop_align,feature_arr[k]);
                print k,result
                if result >= thershold:
                    print 'Found person:',k

