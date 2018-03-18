import _init_paths
import caffe
import cv2
import numpy as np

import numpy as np
import os
import skimage
import sys
import caffe
import math
import sklearn.metrics.pairwise as pw
from fr_wuqianliang_sphereface import *
from matlab_cp2tform import get_similarity_transform_for_cv2

#  load Caffe model

global net;
net = caffe.Classifier('sphereface_deploy.prototxt', 'sphereface_model_iter_28000.caffemodel');

def get_img_crop_align(img):

    # get bbox and lamarks
    boundingboxes, points = get_bbox_and_landmarks(img)
    if len(boundingboxes) == 1:
        landmark = points[0]
#        print landmark
        landmark = [int(landmark[0]),int(landmark[5]),int(landmark[1]),int(landmark[6]),int(landmark[2]),int(landmark[7]),int(landmark[3]),int(landmark[8]),int(landmark[4]),int(landmark[9])]
#	print landmark        
    	img_align_and_crop = alignment(img,landmark)

    	img_align_and_crop = img_align_and_crop.transpose(2, 0, 1).reshape((1,3,112,96))
    	img_align_and_crop = (img_align_and_crop-127.5)/128.0
    	return img_align_and_crop
    return None


def compare_pic(feature1, feature2):
    predicts = pw.cosine_similarity(feature1, feature2);
    return predicts;

def get_feature(img):
    global net;
#    X = read_image(path);
    out = net.forward_all(data = img);
    feature = np.float64(out['fc5']);
    feature = np.reshape(feature, (1, 512));
    return feature;

def alignment(src_img,src_pts):
    ref_pts = [ [30.2946, 51.6963],[65.5318, 51.5014],
        [48.0252, 71.7366],[33.5493, 92.3655],[62.7299, 92.2041] ]
    crop_size = (96,112)
    src_pts = np.array(src_pts).reshape(5,2)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    return face_img


def read_image(filepath):
#    X = np.empty((1,3,112,96));
    filename = filepath.split('\n');
    filename = filename[0];
    img = skimage.io.imread(filename, as_grey=False);
    
    return img;

def get_feature_crop_align(filepath):

    img =  read_image(testFile)
    img=get_img_crop_align(img)
    return get_feature(img)



if __name__ == '__main__':
    thershold = 0.85;
    TEST_SUM = 6000;
   
    TEST_BASE = "./snd_id_photo"; 
    
    testFile = "/home/arthur/sphereface/preprocess/data/lfw/Zinedine_Zidane/Zinedine_Zidane_0001.jpg"; 
    img =  read_image(testFile)
    #print "orginal shape ",img.shape
    img_align = get_img_crop_align(img)
    feature = get_feature(img_align)
    print feature

    # read pic and name
    feature_arr = {}
    for people_path in os.listdir(TEST_BASE):
        ppath = os.path.join(TEST_BASE,people_path)
        for img_name in os.listdir(ppath):
            if img_name == people_path+'.jpg':

                feat = get_feature_crop_align(os.path.join(ppath,img_name))
                if feat is not None:
                    feature_arr[people_path]=feat

    for people_path in os.listdir(TEST_BASE):
        ppath = os.path.join(TEST_BASE,people_path)
        for img_name in os.listdir(ppath):
            if img_name is not people_path+'.jpg':
                feat = get_feature_crop_align(os.path.join(ppath,img_name)) 
                if feat is not None:
                    max_k = ""
                    max_like = 0.0
                    for k in feature_arr:
                        result = compare_pic(feat,feature_arr[k])
                        if result[0][0] > max_like:
                            max_k = k
                            max_like = result[0][0]
                    print people_path+','+img_name+','+max_k+','+str(max_like)
    '''
    cap = cv2.VideoCapture(0)
    start = time()
    while True:

        ret, img = cap.read()

        feature_cam_face_crop_align = get_feature_crop_align_cam(img)
        if feature_cam_face_crop_align is not None:
            for k in feature_arr:
                result = compare_pic(feature_cam_face_crop_align,feature_arr[k]);
                print k,result
                if result >= thershold:
                    print 'Found person:',k
    '''
