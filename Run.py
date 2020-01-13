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

ROOT_DIR = Path('/content/BoundingBox1')
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

    plot_3d_box(img, cam_to_img, orient, dimensions, location) # 3d boxes

    return location

def main():

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
        # print('\n'+img_path+'\n')
        # P for each frame
        # calib_file = calib_path + id + ".txt"

        truth_img = cv2.imread(img_file)
        img = np.copy(truth_img)
        yolo_img = np.copy(truth_img)
        # print(img.shape)
        detections = yolo.detect(yolo_img)
        ampl = 0
        for detection in detections:
            print('\n')
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
                location = plot_regressed_3d_bbox(img, proj_matrix, box_2d, dim, alpha, theta_ray, truth_img)
            else:
                location = plot_regressed_3d_bbox(img, proj_matrix, box_2d, dim, alpha, theta_ray)

            # if not FLAGS.hide_debug:
            #     print('Estimated pose: %s'%location)
            cv2.imwrite('out2/' + 'temp' + ".png", srez)
            img_folder = SAMPLE_DIR
            model_dir = ROOT_DIR
            model_file = 'stage-2-152-c.pkl'
            img_file = 'temp.png'
            pred_class = batch_predict(img_folder, model_dir, model_file,img_file)
            print(classes[int(str(pred_class))])
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
