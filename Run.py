"""
Images must be in ./Kitti/testing/image_2/ and camera matricies in ./Kitti/testing/calib/

Uses YOLO to obtain 2D box, PyTorch to get 3D box, plots both

SPACE bar for next image, any other key to exit
"""
import warnings
warnings.filterwarnings('ignore')
import os
from scipy.io import loadmat
from pathlib import Path
from tqdm import tqdm
import pandas as pd

import fastai
from fastai.vision import *

from torch_lib.Dataset import *
from library.Math import *
from library.Plotting import *
from torch_lib import Model, ClassAverages
from yolo.yolo import cv_Yolo

import os
import time

import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg

import argparse

angleofcam = 40
H = 1600

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()

parser.add_argument("--image-dir", default="eval/image_2/",
                    help="Relative path to the directory containing images to detect. Default \
                    is eval/image_2/")

# TODO: support multiple cal matrix input types
parser.add_argument("--cal-dir", default="camera_cal/",
                    help="Relative path to the directory containing camera calibration form KITTI. \
                    Default is camera_cal/")

parser.add_argument("--video", action="store_true",
                    help="Weather or not to advance frame-by-frame as fast as possible. \
                    By default, this will pull images from ./eval/video")

parser.add_argument("--show-yolo", action="store_true",
                    help="Show the 2D BoundingBox detecions on a separate image")

parser.add_argument("--hide-debug", action="store_true",
                    help="Supress the printing of each 3d location")

ROOT_DIR = Path('')
META_DIR= ROOT_DIR/'devkit'
SAMPLE_DIR = ROOT_DIR/'out2'

def load_classes(meta_dir):
  '''List of classes
  :param meta_dir: 
  :return: idx to class dict
  '''
  classes = loadmat(meta_dir/'cars_meta.mat')
  classes = classes['class_names'][0]
  classes = [y for x in classes for y in x]
  idx_to_class = {idx+1:clss for idx, clss in enumerate(classes)}
  
  return idx_to_class

def batch_predict(img_folder, model_dir, model_file,img_file):
  # preds = {}
  learn = load_learner(model_dir, file=model_file)
  
  # for img_file in tqdm(os.listdir(img_folder)):
  img_path = os.path.join(img_folder, img_file)
  img = open_image(img_path)
  pred_class, _, prob = learn.predict(img)
  # print(classes[int(str(pred_class))])
  # preds[img_file] = (int(str(pred_class)), prob.max().item())
  
  return pred_class

def plot_regressed_3d_bbox(img, cam_to_img, box_2d, dimensions, alpha, theta_ray, img_2d=None):

    # the math! returns X, the corners used for constraint
    location, X = calc_location(dimensions, cam_to_img, box_2d, alpha, theta_ray)

    orient = alpha + theta_ray

    if img_2d is not None:
        plot_2d_box(img_2d, box_2d)

    r,z1, z2 = plot_3d_box(img, cam_to_img, orient, dimensions, location) # 3d boxes

    return location, r, z1,z2

def main():
    df = pd.read_csv('1.csv')
    FLAGS = parser.parse_args()
    classes = load_classes(META_DIR)
    # load torch
    weights_path = os.path.abspath(os.path.dirname(__file__)) + '/weights'
    model_lst = [x for x in sorted(os.listdir(weights_path)) if x.endswith('.pkl')]
    if len(model_lst) == 0:
        print('No previous model found, please train first!')
        exit()
    else:
        print('Using previous model %s'%model_lst[-1])
        my_vgg = vgg.vgg19_bn(pretrained=True)
        # TODO: load bins from file or something
        model = Model.Model(features=my_vgg.features, bins=2).cuda()
        checkpoint = torch.load(weights_path + '/%s'%model_lst[-1])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    print(os.listdir(weights_path))
    # load yolo
    yolo_path = os.path.abspath(os.path.dirname(__file__)) + '/weights'
    yolo = cv_Yolo(yolo_path)

    averages = ClassAverages.ClassAverages()

    # TODO: clean up how this is done. flag?
    angle_bins = generate_bins(2)

    image_dir = FLAGS.image_dir
    cal_dir = FLAGS.cal_dir
    if FLAGS.video:
        if FLAGS.image_dir == "eval/image_2/" and FLAGS.cal_dir == "camera_cal/":
            image_dir = "eval/video/2011_09_26/image_2/"
            cal_dir = "eval/video/2011_09_26/"


    img_path = os.path.abspath(os.path.dirname(__file__)) + "/" + image_dir
    # using P_rect from global calibration file
    calib_path = os.path.abspath(os.path.dirname(__file__)) + "/" + cal_dir
    calib_file = calib_path + "calib_cam_to_cam.txt"

    # using P from each frame
    # calib_path = os.path.abspath(os.path.dirname(__file__)) + '/Kitti/testing/calib/'

    try:
        ids = [x.split('.')[0] for x in sorted(os.listdir(img_path))]
    except:
        print("\nError: no images in %s"%img_path)
        exit()

    for img_id in ids:
        start_time = time.time()

        img_file = img_path + img_id + ".png"
#         print('\n'+img_id+'\n')
        # P for each frame
        # calib_file = calib_path + id + ".txt"
        print(img_file)
        truth_img = cv2.imread(img_file)
        img = np.copy(truth_img)
        yolo_img = np.copy(truth_img)
        # print(img.shape)
        detections = yolo.detect(yolo_img)
        ampl = 0
        lenel = 0
        for detection in detections:
            print('\n')
            lenel+=1
            ampl +=1
            if not averages.recognized_class(detection.detected_class):
                continue

            # this is throwing when the 2d bbox is invalid
            # TODO: better check
            try:
                detectedObject = DetectedObject(img, detection.detected_class, detection.box_2d, calib_file)
            except:
                continue

            theta_ray = detectedObject.theta_ray
            input_img = detectedObject.img
            proj_matrix = detectedObject.proj_matrix
            box_2d = detection.box_2d
            detected_class = detection.detected_class
            procent1 = (-box_2d[0][1]+box_2d[1][1])//5
            procent2 = (-box_2d[0][0]+box_2d[1][0])//5
            if box_2d[0][1]-procent1 <= 0:
              yminim = 0
            else:
              yminim = box_2d[0][1]-procent1
            if box_2d[1][1]+procent1 >= img.shape[0]-1:
              ymaxim = img.shape[0]-1
            else:
              ymaxim = box_2d[1][1]+procent1
            if box_2d[0][0]-procent2 < 0:
              xminim = 0
            else:
              xminim = box_2d[0][0]-procent2
            if box_2d[1][0]+procent2 >= img.shape[1]-1:
              xmaxim = img.shape[1]-1
            else:
              xmaxim = box_2d[1][0]+procent2
            srez = truth_img[yminim:ymaxim, xminim:xmaxim,:]

            # box_2d[]

            input_tensor = torch.zeros([1,3,224,224]).cuda()
            input_tensor[0,:,:,:] = input_img

            [orient, conf, dim] = model(input_tensor)
            orient = orient.cpu().data.numpy()[0, :, :]
            conf = conf.cpu().data.numpy()[0, :]
            dim = dim.cpu().data.numpy()[0, :]

            dim += averages.get_item(detected_class)

            argmax = np.argmax(conf)
            orient = orient[argmax, :]
            cos = orient[0]
            sin = orient[1]
            alpha = np.arctan2(sin, cos)
            alpha += angle_bins[argmax]
            alpha -= np.pi

            if FLAGS.show_yolo:
                location, r, z1, c = plot_regressed_3d_bbox(img, proj_matrix, box_2d, dim, alpha, theta_ray, truth_img)
            else:
                location, r, z1, c = plot_regressed_3d_bbox(img, proj_matrix, box_2d, dim, alpha, theta_ray)

            # if not FLAGS.hide_debug:
            #     print('Estimated pose: %s'%location)
            cv2.imwrite('out2/' + 'temp' + ".png", srez)
            img_folder = SAMPLE_DIR
            model_dir = ROOT_DIR
            model_file = 'stage-2-152-c.pkl'
            img_file = 'temp.png'
            pred_class = batch_predict(img_folder, model_dir, model_file,img_file)
            print('Possible car model:', classes[int(str(pred_class))])
            alpha = angleofcam/img.shape[0] * r/30
            h = df[df['model'] == classes[int(str(pred_class))]].values[0][3]
            if 4*H*(H - h)<=0:
                print("Сan't estimate the distance")
            else:
                if alpha > math.atan(h/math.sqrt(4*H*(H - h))):
                    alpha = math.atan(h/math.sqrt(4*H*(H - h))) - 0.05
    #             print(alpha, math.atan(h/math.sqrt(4*H*(H - h))))
    #             print(img.shape,"&&&&")
                s1 = (h/H + math.sqrt((h*h)/(H*H) - 4*math.tan(alpha)*math.tan(alpha)*(H-h)/H))*H/(2*math.tan(alpha))
                s2 = (h/H - math.sqrt((h*h)/(H*H) - 4*math.tan(alpha)*math.tan(alpha)*(H-h)/H))*H/(2*math.tan(alpha))
                print('Distance to car:',max(s1,s2),'mm')
            a = 0.63
            if img_id == '0000000043':
                if lenel == 1:
                    k = (1 -sqrt((1313*a-z1[0])*(1313*a - z1[0]) + (546*a - z1[1])*(546*a - z1[1]))/c)*100
                    if k>0:
                        print(k,'%')
                    else:
                        print(0,'%')
                elif lenel == 2:
                    k = (1 -sqrt((716*a-z1[0])*(716*a - z1[0]) + (391*a - z1[1])*(391*a - z1[1]))/c)*100
                    if k>0:
                        print(k,'%')
                    else:
                        print(0,'%')
                elif lenel == 3:
                    k = (1 - sqrt((781*a-z1[0])*(781*a - z1[0]) + (341*a - z1[1])*(341*a - z1[1]))/c)*100
                    if k>0:
                        print(k,'%')
                    else:
                        print(0,'%')
                elif lenel == 4:
                    k = (1 -sqrt((1093*a-z1[0])*(1093*a - z1[0]) + (423*a - z1[1])*(423*a - z1[1]))/c)*100
                    if k>0:
                        print(k,'%')
                    else:
                        print(0,'%')
                elif lenel == 5:
                    k = (1 - sqrt((982*a-z1[0])*(982*a - z1[0]) + (348*a - z1[1])*(348*a - z1[1]))/c)*100
                    if k>0:
                        print(k,'%')
                    else:
                        print(0,'%')
            elif img_id == '0000000098':
                if lenel == 1:
                    k = (1 - sqrt((1303*a-z1[0])*(1303*a - z1[0]) + (496*a - z1[1])*(496*a - z1[1]))/c)*100
                    if k>0:
                        print(k,'%')
                    else:
                        print(0,'%')
                elif lenel == 2:
                    k = (1 - sqrt((1180*a-z1[0])*(1180*a - z1[0]) + (406*a - z1[1])*(406*a - z1[1]))/c)*100
                    if k>0:
                        print(k,'%')
                    else:
                        print(0,'%')
                elif lenel == 3:
                    k = (1 - sqrt((761*a-z1[0])*(761*a - z1[0]) + (399*a - z1[1])*(399*a - z1[1]))/c)*100
                    if k>0:
                        print(k,'%')
                    else:
                        print(0,'%')
                elif lenel == 4:
                    k = (1 - sqrt((657*a-z1[0])*(657*a - z1[0]) + (461*a - z1[1])*(461*a - z1[1]))/c)*100
                    if k>0:
                        print(k,'%')
                    else:
                        print(0,'%')
                elif lenel == 5:
                    k = (1 - sqrt((898*a-z1[0])*(898*a - z1[0]) + (312*a - z1[1])*(312*a - z1[1]))/c)*100
                    if k>0:
                        print(k,'%')
                    else:
                        print(0,'%')
            elif img_id == '0000000191':
                if lenel == 1:
                    k = (1 - sqrt((1212*a-z1[0])*(1212*a - z1[0]) + (449*a - z1[1])*(449*a - z1[1]))/c)*100
                    if k>0:
                        print(k,'%')
                    else:
                        print(0,'%')
                elif lenel == 2:
                    k = (1 - sqrt((757*a-z1[0])*(757*a - z1[0]) + (398*a - z1[1])*(398*a - z1[1]))/c)*100
                    if k>0:
                        print(k,'%')
                    else:
                        print(0,'%')
                elif lenel == 3:
                    k = (1 - sqrt((838*a-z1[0])*(838*a - z1[0]) + (335*a - z1[1])*(335*a - z1[1]))/c)*100
                    if k>0:
                        print(k,'%')
                    else:
                        print(0,'%')
                elif lenel == 4:
                    k = (1 - sqrt((1129*a-z1[0])*(1129*a - z1[0]) + (395*a - z1[1])*(395*a - z1[1]))/c)*100
                    if k>0:
                        print(k,'%')
                    else:
                        print(0,'%')
                elif lenel == 5:
                    k = (1 - sqrt((980*a-z1[0])*(980*a - z1[0]) + (325*a - z1[1])*(325*a - z1[1]))/c)*100
                    if k>0:
                        print(k,'%')
                    else:
                        print(0,'%')
            elif img_id == '0000000272':
                k = (1 - sqrt((1251*a-z1[0])*(1251*a - z1[0]) + (467*a - z1[1])*(467*a - z1[1]))/c)*100
                if k>0:
                    print(k,'%')
                else:
                    print(0,'%')
            elif img_id == '0000000559':
                if lenel == 1:
                    k = (1 - sqrt((753*a-z1[0])*(753*a - z1[0]) + (400*a - z1[1])*(400*a - z1[1]))/c)*100
                    if k>0:
                        print(k,'%')
                    else:
                        print(0,'%')
                elif lenel == 2:
                    k = (1 - sqrt((815*a-z1[0])*(815*a - z1[0]) + (366*a - z1[1])*(366*a - z1[1]))/c)*100
                    if k>0:
                        print(k,'%')
                    else:
                        print(0,'%')
            elif img_id == '0000000722':
                if lenel == 1:
                    k = (1 - sqrt((1395*a-z1[0])*(1395*a - z1[0]) + (508*a- z1[1])*(508*a - z1[1]))/c)*100
                    if k>0:
                        print(k,'%')
                    else:
                        print(0,'%')
                elif lenel == 2:
                    k = (1 - sqrt((849*a-z1[0])*(849*a - z1[0]) + (377*a - z1[1])*(377*a - z1[1]))/c)*100
                    if k>0:
                        print(k,'%')
                    else:
                        print(0,'%')
                elif lenel == 3:
                    k = (1 - sqrt((1035*a-z1[0])*(1035*a - z1[0]) + (360*a - z1[1])*(360*a - z1[1]))/c)*100
                    if k>0:
                        print(k,'%')
                    else:
                        print(0,'%')
            elif img_id == '0000001038':
                if lenel == 1:
                    k = (1 - sqrt((371*a-z1[0])*(371*a - z1[0]) + (418*a - z1[1])*(418*a - z1[1]))/c)*100
                    if k>0:
                        print(k,'%')
                    else:
                        print(0,'%')
                elif lenel == 2:
                    k = (1 - sqrt((812*a-z1[0])*(812*a - z1[0]) + (366*a - z1[1])*(366*a - z1[1]))/c)*100
                    if k>0:
                        print(k,'%')
                    else:
                        print(0,'%')
                elif lenel == 3:
                    k = (1 - sqrt((900*a-z1[0])*(900*a - z1[0]) + (316*a - z1[1])*(316*a - z1[1]))/c)*100
                    if k>0:
                        print(k,'%')
                    else:
                        print(0,'%')
                elif lenel == 4:
                    k = (1 - sqrt((1141*a -z1[0])*(1141*a - z1[0]) + (489*a - z1[1])*(489*a - z1[1]))/c)*100
                    if k>0:
                        print(k,'%')
                    else:
                        print(0,'%')
            elif img_id == '1000000529':
                if lenel == 1:
                    k = (1 - sqrt((563*a-z1[0])*(563*a - z1[0]) + (462*a - z1[1])*(462*a - z1[1]))/c)*100
                    if k>0:
                        print(k,'%')
                    else:
                        print(0,'%')
                elif lenel == 2:
                    k = (1 - sqrt((685*a-z1[0])*(685*a - z1[0]) + (405*a - z1[1])*(405*a - z1[1]))/c)*100
                    if k>0:
                        print(k,'%')
                    else:
                        print(0,'%')
            elif img_id == '1000000594':
                if lenel == 1:
                    k = (1 - sqrt((648*a-z1[0])*(648*a - z1[0]) + (470*a - z1[1])*(470*a - z1[1]))/c)*100
                    if k>0:
                        print(k,'%')
                    else:
                        print(0,'%')
                elif lenel == 2:
                    k = (1 - sqrt((796*a-z1[0])*(796*a - z1[0]) + (359*a - z1[1])*(359*a - z1[1]))/c)*100
                    if k>0:
                        print(k,'%')
                    else:
                        print(0,'%')
                elif lenel == 3:
                    k = (1 - sqrt((1268*a-z1[0])*(1268*a - z1[0]) + (458*a - z1[1])*(458*a - z1[1]))/c)*100
                    if k>0:
                        print(k,'%')
                    else:
                        print(0,'%')
                elif lenel == 4:
                    k = (1 - sqrt((1071*a-z1[0])*(1071*a - z1[0]) + (352*a - z1[1])*(352*a - z1[1]))/c)*100
                    if k>0:
                        print(k,'%')
                    else:
                        print(0,'%')
            elif img_id == '1000000616':
                if lenel == 1:
                    k = (1 - sqrt((1319*a-z1[0])*(1319*a - z1[0]) + (489*a - z1[1])*(489*a - z1[1]))/c)*100
                    if k>0:
                        print(k,'%')
                    else:
                        print(0,'%')
                elif lenel == 2:
                    k = (1 - sqrt((1021*a-z1[0])*(1021*a - z1[0]) + (313*a - z1[1])*(313*a - z1[1]))/c)*100
                    if k>0:
                        print(k,'%')
                    else:
                        print(0,'%')
            elif img_id == '1000000776':
                if lenel == 1:
                    k = (1 - sqrt((1458*a-z1[0])*(1458*a - z1[0]) + (446*a - z1[1])*(446*a - z1[1]))/c)*100
                    if k>0:
                        print(k,'%')
                    else:
                        print(0,'%')
                elif lenel == 2:
                    k = (1 - sqrt((1160*a-z1[0])*(1160*a - z1[0]) + (373*a - z1[1])*(373*a - z1[1]))/c)*100
                    if k>0:
                        print(k,'%')
                    else:
                        print(0,'%')
                elif lenel == 3:
                    k = (1 - sqrt((656*a-z1[0])*(656*a - z1[0]) + (359*a - z1[1])*(359*a - z1[1]))/c)*100
                    if k>0:
                        print(k,'%')
                    else:
                        print(0,'%')
                elif lenel == 4:
                    k = (1 - sqrt((1103*a -z1[0])*(1103*a - z1[0]) + (349*a - z1[1])*(349*a - z1[1]))/c)*100
                    if k>0:
                        print(k,'%')
                    else:
                        print(0,'%')
                elif lenel == 5:
                    k = (1 - sqrt((930*a-z1[0])*(930*a - z1[0]) + (335*a - z1[1])*(335*a - z1[1]))/c)*100
                    if k>0:
                        print(k,'%')
                    else:
                        print(0,'%')
                elif lenel == 6:
                    k = (1 - sqrt((1036*a-z1[0])*(1036*a - z1[0]) + (340*a - z1[1])*(340*a - z1[1]))/c)*100
                    if k>0:
                        print(k,'%')
                    else:
                        print(0,'%')
                elif lenel == 7:
                    k = (1 - sqrt((584*a -z1[0])*(584*a - z1[0]) + (386*a - z1[1])*(386*a - z1[1]))/c)*100
                    if k>0:
                        print(k,'%')
                    else:
                        print(0,'%')
                elif lenel == 8:
                    k = (1 - sqrt((435*a-z1[0])*(435*a - z1[0]) + (426*a - z1[1])*(426*a - z1[1]))/c)*100
                    if k>0:
                        print(k,'%')
                    else:
                        print(0,'%')
                elif lenel == 9:
                    k = (1 - sqrt((754*a -z1[0])*(754*a - z1[0]) + (328*a - z1[1])*(328*a - z1[1]))/c)*100
                    if k>0:
                        print(k,'%')
                    else:
                        print(0,'%')
            elif img_id == '1000000864':
                if lenel == 1:
                    k = (1 - sqrt((1651*a-z1[0])*(1651*a - z1[0]) + (463*a - z1[1])*(463*a - z1[1]))/c)*100
                    if k>0:
                        print(k,'%')
                    else:
                        print(0,'%')
                elif lenel == 2:
                    k = (1 - sqrt((1031*a-z1[0])*(1031*a - z1[0]) + (302*a - z1[1])*(302*a - z1[1]))/c)*100
                    if k>0:
                        print(k,'%')
                    else:
                        print(0,'%')
                elif lenel == 3:
                    k = (1 - sqrt((962*a-z1[0])*(962*a - z1[0]) + (348*a - z1[1])*(348*a - z1[1]))/c)*100
                    if k>0:
                        print(k,'%')
                    else:
                        print(0,'%')
                elif lenel == 4:
                    k = (1 - sqrt((655*a -z1[0])*(655*a - z1[0]) + (355*a - z1[1])*(355*a - z1[1]))/c)*100
                    if k>0:
                        print(k,'%')
                    else:
                        print(0,'%')
                elif lenel == 5:
                    k = (1 - sqrt((802*a-z1[0])*(802*a - z1[0]) + (310*a - z1[1])*(310*a - z1[1]))/c)*100
                    if k>0:
                        print(k,'%')
                    else:
                        print(0,'%')
        if FLAGS.show_yolo:
            numpy_vertical = np.concatenate((truth_img, img), axis=0)
            cv2.imwrite('out/'+img_id + ".png", numpy_vertical)
            # cv2.imshow('SPACE for next image, any other key to exit', numpy_vertical)
        else:
            cv2.imwrite('out/'+img_id + ".png", img)
            # cv2.imshow('3D detections', img)
        
        if not FLAGS.hide_debug:
            print("\n")
            print('Got %s poses in %.3f seconds'%(len(detections), time.time() - start_time))
            print('-------------')

        # if FLAGS.video:
        #     cv2.waitKey(1)
        # else:
        #     if cv2.waitKey(0) != 32: # space bar
        #         exit()

if __name__ == '__main__':
    main()
