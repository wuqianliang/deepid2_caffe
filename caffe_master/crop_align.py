#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import sys
sys.path.append("../test_lfw/")
from fr_wuqianliang import *
import random
import cv2

##############################################
#读取clean list
clean_list = {}
file_object = open('MS-Celeb-1M_clean_list.txt', 'r')
for line in file_object:
    arr=line.split(' ')
    clean_list[arr[0]]=1
##############################################

new_folder="/media/arthur/snd_drive/test"
def walk_through_folder_for_split(src_folder):
    test_set  = []
    train_set = []
    global process_count
    label = 0
    for people_folder in os.listdir(src_folder):
        people_path = src_folder+people_folder + '/'
        img_files = os.listdir(people_path)
        people_imgs = []
        for img_file in img_files:
            img_path = people_folder+'/' + img_file
            #########################################################
            img = cv2.imread(people_path+'/'+img_file)
            crop_image = crop_align_image(img)
            if clean_list.has_key(img_path) and crop_image is not None:
                isExists=os.path.exists(new_folder+'/'+people_folder)
                if not isExists:
                    os.makedirs(new_folder+'/'+people_folder)
                #print img_path,new_folder+'/'+people_folder+'/'+img_file
                cv2.imwrite(new_folder+'/'+people_folder+'/'+img_file, crop_image);
            #########################################################
                people_imgs.append((img_path, label))
                if len(people_imgs) > 10:
                    break;

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: python %s src_folder' % (sys.argv[0]))
        sys.exit()
    src_folder     = sys.argv[1]
    walk_through_folder_for_split(src_folder)
