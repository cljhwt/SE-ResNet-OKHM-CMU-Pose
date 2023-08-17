import unittest
import os
import torch
import sys

sys.path.append('/home/caffe/wangcl/pytorch_Realtime_Multi-Person_Pose_Estimation-master/')
from evaluate.coco_eval import run_eval
from network.rtpose_vgg import get_model, use_vgg, get_mobileNet_model, use_MobileNet, Net_other, get_model_res
from torch import load

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Notice, if you using the
with torch.autograd.no_grad():
    # this path is with respect to the root of the project
    # weight_name = './network/weight/pose_model_scratch.pth'
    # weight_name = '/home/data/wangchenliang/pytorch-paf-model/trained_model/ResNetXt-Net/best_pose-own-rsext-164.pth'
    weight_name = '/home/data/wangchenliang/pytorch-paf-model/pretrained-MobileNet-only-okhm-start-100/best_pose-pretrained-MobileNet-200.pth'
    state_dict = torch.load(weight_name)
    # print(state_dict)
    # model = get_model(trunk='vgg19')
    model = get_mobileNet_model()

    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(state_dict)
    model.eval()
    model.float()
    model = model.cuda()

    # The choice of image preprocessing include: 'rtpose', 'inception', 'vgg' and 'ssd'.
    # If you use the converted model from caffe, it is 'rtpose' preprocess, the model trained in
    # this repo used 'vgg' preprocess

    # run_eval(image_dir= '/data/coco/images/', anno_dir = '/data/coco', vis_dir = '/data/coco/vis',
    #     image_list_txt='./evaluate/image_info_val2014_1k.txt',
    #     model=model, preprocess='vgg')

    run_eval(image_dir='/home/caffe/wangcl/wangcl/openpose/tests_database/val_images', anno_dir='/data/coco',
             vis_dir='/home/data/wangchenliang/pytorch-paf-model/pretrained-MobileNet-only-okhm-start-100/vis-200',
             image_list_txt='./evaluate/image_info_val2014_1k.txt',
             model=model, preprocess='vgg')


