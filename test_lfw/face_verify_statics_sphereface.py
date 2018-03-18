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

    #print "1",img.shape
    # get bbox and lamarks
    boundingboxes, points = get_bbox_and_landmarks(img)
    if len(boundingboxes) == 1:
        landmark = points[0]
#        print landmark
        landmark = [int(landmark[0]),int(landmark[5]),int(landmark[1]),int(landmark[6]),int(landmark[2]),int(landmark[7]),int(landmark[3]),int(landmark[8]),int(landmark[4]),int(landmark[9])]
#	print landmark       
    #	print "2",img.shape
    	img_align_and_crop = alignment(img,landmark)
    #    print "img_align_and_crop",img_align_and_crop.shape
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

    img1= read_image(filepath)
    img2=get_img_crop_align(img1)
    if img2 is None:
        print filepath,"==============="
	return None
    return get_feature(img2)



if __name__ == '__main__':
    thershold = 0.85;
    TEST_SUM = 6000;
   
    TEST_BASE = "./snd_id_photo"; 
    
    #testFile = "/home/arthur/sphereface/preprocess/data/lfw/Zinedine_Zidane/Zinedine_Zidane_0001.jpg"; 
    #img =  read_image(testFile)
    #print "orginal shape ",img.shape
    #img_align = get_img_crop_align(img)
    #feature = get_feature(img_align)
    #print feature
    # read pic and name
    feature_arr = {}
    for people_path in os.listdir(TEST_BASE):
        ppath = os.path.join(TEST_BASE,people_path)
        for img_name in os.listdir(ppath):
            if img_name == people_path+'.jpg':
                feat = get_feature_crop_align(os.path.join(ppath,img_name))
                if feat is not None:
        #            print feat
                    feature_arr[people_path]=feat
    for people_path in os.listdir(TEST_BASE):
        ppath = os.path.join(TEST_BASE,people_path)
        for img_name in os.listdir(ppath):
            if img_name is not people_path+'.jpg':
                print os.path.join(ppath,img_name)
                feat = get_feature_crop_align(os.path.join(ppath,img_name))
                if feat is not None:
                    for k in feature_arr:
			f1 = np.array(feat[0])
 			f2=np.array(feature_arr[k][0])

                        #print f1,f2
                        lx = np.sqrt(f1.dot(f1))
			ly = np.sqrt(f2.dot(f2))
			cos =f1.dot(f2)/(lx*ly)
                    	print os.path.join(ppath,img_name)+','+k+','+str(cos)
