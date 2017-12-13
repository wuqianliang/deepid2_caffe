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

#  sys.path.insert(0, '/Downloads/caffe-master/python');
#  load Caffe model

caffe.set_mode_gpu();

global net;
net = caffe.Classifier('DeepID2_deploy.prototxt', 'deepid2_iter_30000.caffemodel');

def compare_pic(feature1, feature2):
    predicts = pw.cosine_similarity(feature1, feature2);
    return predicts;

def get_feature(path, mean_blob):
    global net;
    X = read_image(path, mean_blob);
    # test_num = np.shape(X)[0];
    # print test_num;
    out = net.forward_all(data = X);
    feature = np.float64(out['fc160']);
    feature = np.reshape(feature, (1, 160));
    return feature;

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
    DATA_BASE = "/home/arthur/caffe-master/data/lfw/";
    MEAN_FILE = 'DeepID2_mean.proto';
    POSITIVE_TEST_FILE = "positive_pairs_path.txt";
    NEGATIVE_TEST_FILE = "negative_pairs_path.txt";
    
    mean_blob = caffe.proto.caffe_pb2.BlobProto();
    mean_blob.ParseFromString(open(MEAN_FILE, 'rb').read());
    mean_npy = caffe.io.blobproto_to_array(mean_blob);
    
    #print mean_npy.shape
    
    for thershold in np.arange(0.25, 0.35, 0.01):
        True_Positive = 0;
        True_Negative = 0;
    	False_Positive = 0;
    	False_Negative = 0;

        # Positive Test
        f_positive = open(POSITIVE_TEST_FILE, "r");
        PositiveDataList = f_positive.readlines(); 
        f_positive.close( );
        f_negative = open(NEGATIVE_TEST_FILE, "r");
        NegativeDataList = f_negative.readlines(); 
        f_negative.close( );
        for index in range(len(PositiveDataList)):
            filepath_1 = PositiveDataList[index].split(' ')[0];
            filepath_2 = PositiveDataList[index].split(' ')[1][:-2];
            feature_1 = get_feature(DATA_BASE + filepath_1, mean_npy);
            feature_2 = get_feature(DATA_BASE + filepath_2, mean_npy);
            result = compare_pic(feature_1, feature_2);
            if result >= thershold:
                #  print 'Same Guy\n\n'
                True_Positive += 1;
            else:
                #  wrong
                False_Positive += 1;
                
        for index in range(len(NegativeDataList)):
            filepath_1 = NegativeDataList[index].split(' ')[0];
            filepath_2 = NegativeDataList[index].split(' ')[1][:-2];
            feature_1 = get_feature(DATA_BASE + filepath_1, mean_npy);
            feature_2 = get_feature(DATA_BASE + filepath_2, mean_npy);

            result = compare_pic(feature_1, feature_2);
            if result >= thershold:
                print 'Wrong Guy\n'
                print filepath_1,filepath_2
                #  wrong
                False_Negative += 1;
            else:
                #  correct
                print 'Correct\n'
                print filepath_1,filepath_2
                True_Negative += 1; 

        print "thershold: " + str(thershold);
        print "Accuracy: " + str(float(True_Positive + True_Negative)/TEST_SUM) + " %";
        print "True_Positive: " + str(float(True_Positive)/TEST_SUM) + " %";
        print "True_Negative: " + str(float(True_Negative)/TEST_SUM) + " %";
        print "False_Positive: " + str(float(False_Positive)/TEST_SUM) + " %";
        print "False_Negative: " + str(float(False_Negative)/TEST_SUM) + " %";
