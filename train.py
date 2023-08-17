import argparse
import time
import os
import numpy as np
from collections import OrderedDict
import cv2

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

#import encoding
from network.rtpose_vgg import get_model, use_vgg, use_SEResNet, get_mobileNet_model, use_MobileNet, Net_other, get_model_res
from training.datasets.coco import get_loader

# Hyper-params
parser = argparse.ArgumentParser(description='PyTorch rtpose Training')
# parser.add_argument('--data_dir', default='./data/coco/images', type=str, metavar='DIR',
#                     help='path to where coco images stored')
parser.add_argument('--init_model_param', default='/home/data/wangchenliang/pytorch-paf-model/pretrained-MobileNet-okhm-kd/init_model_param.pth', type=str)
parser.add_argument('--tmodel_path_file', default='/home/data/wangchenliang/pytorch-paf-model/paper-model/pose_model_scratch.pth', type=str)
parser.add_argument('--data_dir', default='/home/data/wangchenliang/COCO2014/data/coco/images', type=str, metavar='DIR',
                    help='path to where coco images stored')
# parser.add_argument('--mask_dir', default='./data/coco/', type=str, metavar='DIR',
#                     help='path to where coco images stored')
parser.add_argument('--mask_dir', default='/home/data/wangchenliang/COCO2014/data/coco/', type=str, metavar='DIR',
                    help='path to where coco images stored')
parser.add_argument('--logdir', default='/logs', type=str, metavar='DIR',
                    help='path to where tensorboard log restore')     #log信息存储位置,/home/data/wangchenliang/pytorch-paf-model/trained_model/ResNetXt-Net/tensorboy
# parser.add_argument('--json_path', default='./data/coco/COCO.json', type=str, metavar='PATH',
#                     help='path to where coco images stored')   #json标注文件存储位置
parser.add_argument('--json_path', default='/home/data/wangchenliang/COCO2014/data/coco/COCO.json', type=str, metavar='PATH',
                    help='path to where coco images stored')   #json标注文件存储位置

# parser.add_argument('--model_path', default='./network/weight/', type=str, metavar='DIR',
#                     help='path to where the model saved') #生成的网络模型保存位置
parser.add_argument('--model_path', default='/home/data/wangchenliang/pytorch-paf-model/ori_model/', type=str, metavar='DIR',
                    help='path to where the model saved') #预处理的网络模型保存位置
                    
parser.add_argument('--lr', '--learning-rate', default=1., type=float,
                    metavar='LR', help='initial learning rate') #学习率设置

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
 
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')#完整训练集训练次数
# parser.add_argument('--epochs', default=1, type=int, metavar='N',
#                     help='number of total epochs to run')#完整训练集训练次数
                    
parser.add_argument('--weight-decay', '--wd', default=0.000, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')  #权重衰减率,正则化项前的一个系数
parser.add_argument('--nesterov', dest='nesterov', action='store_true')   #优化方法
                                                   
parser.add_argument('-o', '--optim', default='sgd', type=str) #sgd方法
#Device options
# parser.add_argument('--gpu_ids', dest='gpu_ids', help='which gpu to use', nargs="+",
#                     default=[0,1,2,3], type=int) #nargs="+"表示至少一个
parser.add_argument('--gpu_ids', dest='gpu_ids', help='which gpu to use', nargs="+",
                    default=[2, 3], type=int) #nargs="+"表示至少一个
                    
parser.add_argument('-b', '--batch_size', default=23, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
# parser.add_argument('-b', '--batch_size', default=1, type=int,
#                     metavar='N', help='mini-batch size (default: 256)')

parser.add_argument('--print_freq', default=20, type=int, metavar='N',
                    help='number of iterations to print the training statistics') #多少次迭代后打印
# parser.add_argument('--print_freq', default=1, type=int, metavar='N',
#                     help='number of iterations to print the training statistics') #多少次迭代后打印
from tensorboardX import SummaryWriter      
args = parser.parse_args()  
               
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in args.gpu_ids)

params_transform = dict()
params_transform['mode'] = 5
# === aug_scale ===
params_transform['scale_min'] = 0.5
params_transform['scale_max'] = 1.1
params_transform['scale_prob'] = 1
params_transform['target_dist'] = 0.6
# === aug_rotate ===
params_transform['max_rotate_degree'] = 40

# === 干扰 ===
params_transform['center_perterb_max'] = 40

# === aug_flip ===
params_transform['flip_prob'] = 0.5

params_transform['np'] = 56 #18+38
params_transform['sigma'] = 7.0
params_transform['limb_width'] = 1.


# prepare for refine loss
def ohkm(loss, top_k):
    ohkm_loss = 0.
    #loss = loss[:, :18]
    for i in range(loss.size()[0]):  # batch_size的个数
        sub_loss = loss[i]  # 17*1*1
        topk_val, topk_idx = torch.topk(sub_loss, k=top_k, dim=0, sorted=False)
        # print(topk_val)
        # print(topk_idx)
        tmp_loss = torch.gather(sub_loss, 0, topk_idx) #将找出的topk_idx放到一起
        ohkm_loss += torch.sum(tmp_loss) / top_k #单幅图像top_k个关节点取均值
    ohkm_loss /= loss.size()[0] #batch_size个输入取均值
    return ohkm_loss

def build_names(): #loss_stage(1-6)_L(1-2)
    names = []

    for j in range(1, 5):
        for k in range(1, 3):
            names.append('loss_stage%d_L%d' % (j, k))
    return names

def get_loss(saved_for_loss, heat_temp_total, heat_weight,
               vec_temp, vec_weight, NetHeatmap, NetPAF):
# def get_loss(saved_for_loss, heat_temp, heat_weight,
#                vec_temp, vec_weight):

    heat_temp_0 = heat_temp_total[0]
    heat_temp_1 = heat_temp_total[1]
    heat_temp_2 = heat_temp_total[2]
    heat_temp_3 = heat_temp_total[3]

    names = build_names()
    saved_for_log = OrderedDict()
    criterion = nn.MSELoss(size_average=True).cuda() #for Global loss
    criterion1 = nn.MSELoss(reduce=False).cuda()  # for refine loss
    #criterion = encoding.nn.DataParallelCriterion(criterion, device_ids=args.gpu_ids)
    total_loss = 0

    #标注区域的loss
    for j in range(4):

        if j == 0:
            pred1 = saved_for_loss[2 * j] * vec_weight
            """
            print("pred1 sizes")
            print(saved_for_loss[2*j].data.size())
            print(vec_weight.data.size())
            print(vec_temp.data.size())
            """
            gt1 = vec_temp * vec_weight
            tlabel1 = NetPAF * vec_weight
            #未标注部分的PAF标签
            untlabel1 = NetPAF * (1 - vec_weight)

            pred2 = saved_for_loss[2 * j + 1] * heat_weight
            gt2 = heat_weight * heat_temp_0
            tlabel2 = NetHeatmap * heat_weight
            # 未标注部分的Heatmap标签
            untlabel2 = NetHeatmap * (1 - heat_weight)

            """
            print("pred2 sizes")
            print(saved_for_loss[2*j+1].data.size())
            print(heat_weight.data.size())
            print(heat_temp.data.size())
            """
            #print('stage0')

        elif j == 1:
            pred1 = saved_for_loss[2 * j] * vec_weight
            """
            print("pred1 sizes")
            print(saved_for_loss[2*j].data.size())
            print(vec_weight.data.size())
            print(vec_temp.data.size())
            """
            gt1 = vec_temp * vec_weight
            tlabel1 = NetPAF * vec_weight
            # 未标注部分的PAF标签
            untlabel1 = NetPAF * (1 - vec_weight)

            pred2 = saved_for_loss[2 * j + 1] * heat_weight
            gt2 = heat_weight * heat_temp_1
            tlabel2 = NetHeatmap * heat_weight
            # 未标注部分的Heatmap标签
            untlabel2 = NetHeatmap * (1 - heat_weight)
            """
            print("pred2 sizes")
            print(saved_for_loss[2*j+1].data.size())
            print(heat_weight.data.size())
            print(heat_temp.data.size())
            """
            #print('stage1')
        elif j == 2:
            pred1 = saved_for_loss[2 * j] * vec_weight
            """
            print("pred1 sizes")
            print(saved_for_loss[2*j].data.size())
            print(vec_weight.data.size())
            print(vec_temp.data.size())
            """
            gt1 = vec_temp * vec_weight
            tlabel1 = NetPAF * vec_weight
            # 未标注部分的PAF标签
            untlabel1 = NetPAF * (1 - vec_weight)

            pred2 = saved_for_loss[2 * j + 1] * heat_weight
            gt2 = heat_weight * heat_temp_2
            tlabel2 = NetHeatmap * heat_weight
            # 未标注部分的Heatmap标签
            untlabel2 = NetHeatmap * (1 - heat_weight)
            """
            print("pred2 sizes")
            print(saved_for_loss[2*j+1].data.size())
            print(heat_weight.data.size())
            print(heat_temp.data.size())
            """
            #print('stage2')
        else:
            pred1 = saved_for_loss[2 * j] * vec_weight
            """
            print("pred1 sizes")
            print(saved_for_loss[2*j].data.size())
            print(vec_weight.data.size())
            print(vec_temp.data.size())
            """
            gt1 = vec_temp * vec_weight
            tlabel1 = NetPAF * vec_weight
            # 未标注部分的PAF标签
            untlabel1 = NetPAF * (1 - vec_weight)

            pred2 = saved_for_loss[2 * j + 1] * heat_weight
            gt2 = heat_weight * heat_temp_3
            tlabel2 = NetHeatmap * heat_weight
            # 未标注部分的Heatmap标签
            untlabel2 = NetHeatmap * (1 - heat_weight)

            """
            print("pred2 sizes")
            print(saved_for_loss[2*j+1].data.size())
            print(heat_weight.data.size())
            print(heat_temp.data.size())
            """
            #print('stage3-5')

        # pred1 = saved_for_loss[2 * j] * vec_weight #1*38*46*46
        # """
        # print("pred1 sizes")
        # print(saved_for_loss[2*j].data.size())
        # print(vec_weight.data.size())
        # print(vec_temp.data.size())
        # """
        # gt1 = vec_temp * vec_weight
        #
        # pred2 = saved_for_loss[2 * j + 1] * heat_weight
        # gt2 = heat_weight * heat_temp
        # """
        # print("pred2 sizes")
        # print(saved_for_loss[2*j+1].data.size())
        # print(heat_weight.data.size())
        # print(heat_temp.data.size())
        # """

        # Compute losses
        #标注部分PAF的loss
        loss1 = criterion(pred1,  gt1)
        kdloss1 = criterion(pred1, tlabel1)
        #未标注部分PAF的loss
        unkdloss1 = criterion(pred1, untlabel1)
        #loss2 = criterion(pred2, gt2)
        #loss2 = criterion(pred2, gt2)

        #HeatMap的loss
        if j <= 2: #Global loss 0-2 stage
            #标注部分HeatMap的loss
            loss2 = criterion(pred2, gt2)
            kdloss2 = criterion(pred2, tlabel2)
            #未标注部分HeatMap的loss
            unkdloss2 = criterion(pred2, untlabel2)

            #print ('Global loss 0-2 stage')

        else: #Refine loss 3 stage
            #loss3 = criterion(pred2, gt2)
            # print('loss3')
            # print(loss3)
            # 标注部分HeatMap的loss
            loss2 = criterion1(pred2, gt2)
            loss2 = loss2.mean(dim=3).mean(dim=2)
            #ori:8
            loss2 = ohkm(loss2, 8)

            kdloss2 = criterion1(pred2, tlabel2)
            kdloss2 = kdloss2.mean(dim=3).mean(dim=2)
            # ori:8
            kdloss2 = ohkm(kdloss2, 8)

            unkdloss2 = criterion1(pred2, untlabel2)
            unkdloss2 = unkdloss2.mean(dim=3).mean(dim=2)
            # ori:8
            unkdloss2 = ohkm(unkdloss2, 8)
            # print('loss2')
            # print(loss2)
            #print('Refine loss 0-3 stage')

        kdloss_alpha = 0.5
        unkdloss_alpha = 1
        # total_loss += (1 - kdloss_alpha) * loss1 + kdloss_alpha * kdloss1 + unkdloss_alpha * unkdloss1
        # total_loss += (1 - kdloss_alpha) * loss2 + kdloss_alpha * kdloss2 + unkdloss_alpha * unkdloss2
        total_loss += (1 - kdloss_alpha) * loss1 + kdloss_alpha * kdloss1
        total_loss += (1 - kdloss_alpha) * loss2 + kdloss_alpha * kdloss2
        # total_loss += unkdloss_alpha * unkdloss1
        # total_loss += unkdloss_alpha * unkdloss2
        #print(total_loss)
        #print(unkdloss_alpha * unkdloss1)

        # Get value from Variable and save for log
        saved_for_log[names[2 * j]] = (1 - kdloss_alpha) * loss1.item() + kdloss_alpha * kdloss1.item()
        saved_for_log[names[2 * j + 1]] = (1 - kdloss_alpha) * loss2.item() + kdloss_alpha * kdloss2.item()

    saved_for_log['max_ht'] = torch.max(
        saved_for_loss[-1].data[:, 0:-1, :, :]).item()
    saved_for_log['min_ht'] = torch.min(
        saved_for_loss[-1].data[:, 0:-1, :, :]).item()
    saved_for_log['max_paf'] = torch.max(saved_for_loss[-2].data).item()
    saved_for_log['min_paf'] = torch.min(saved_for_loss[-2].data).item()

    return total_loss, saved_for_log
         

def train(train_loader, tmodel, model, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    meter_dict = {}
    for name in build_names():
        meter_dict[name] = AverageMeter()
    meter_dict['max_ht'] = AverageMeter()
    meter_dict['min_ht'] = AverageMeter()    
    meter_dict['max_paf'] = AverageMeter()    
    meter_dict['min_paf'] = AverageMeter()
    
    
    # switch to train mode
    model.train()

    end = time.time()
    for i, (img, heatmap_target_total, heat_mask, paf_target, paf_mask) in enumerate(train_loader):
    #for i, (img, heatmap_target, heat_mask, paf_target, paf_mask) in enumerate(train_loader):
        # measure data loading time
        #writer.add_text('Text', 'text logged at step:' + str(i), i)
        
        #for name, param in model.named_parameters():
        #    writer.add_histogram(name, param.clone().cpu().data.numpy(),i)        
        data_time.update(time.time() - end)

        # cv2.imshow('input',img)
        # cv2.waitKey(0)

        img = img.cuda()
        #heatmap_target = heatmap_target.cuda()

        heatmap_target_total_CUDA = []
        heatmap_target_total_CUDA.append(heatmap_target_total[0].cuda())
        heatmap_target_total_CUDA.append(heatmap_target_total[1].cuda())
        heatmap_target_total_CUDA.append(heatmap_target_total[2].cuda())
        heatmap_target_total_CUDA.append(heatmap_target_total[3].cuda())

        heat_mask = heat_mask.cuda()
        #print(heat_mask)
        #print(unlabeled_heat_mask)
        paf_target = paf_target.cuda()
        paf_mask = paf_mask.cuda()

        # compute output
        _,saved_for_loss = model(img)#网络前向运行

        #教师网络
        NetOutput, _ = tmodel(img)
        NetPAF = NetOutput[-2]
        NetHeatmap = NetOutput[-1]
        # print('NetPAF')
        # print(NetPAF.shape)
        # print('NetHeatmap')
        # print(NetHeatmap.shape)

        # total_loss, saved_for_log = get_loss(saved_for_loss, heatmap_target, heat_mask,
        #        paf_target, paf_mask)
        total_loss, saved_for_log = get_loss(saved_for_loss, heatmap_target_total_CUDA, heat_mask,
                                            paf_target, paf_mask, NetHeatmap, NetPAF)
        
        for name,_ in meter_dict.items():
            meter_dict[name].update(saved_for_log[name], img.size(0))
        losses.update(total_loss, img.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()#进入下一个mini_batch

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print_string = 'Epoch: [{0}][{1}/{2}]\t'.format(epoch, i, len(train_loader))
            print_string +='Data time {data_time.val:.3f} ({data_time.avg:.3f})\n'.format( data_time=data_time)
            print_string += 'Loss {loss.val:.4f} ({loss.avg:.4f})\n'.format(loss=losses)

            for name, value in meter_dict.items():
                print_string+='{name}: {loss.val:.4f} ({loss.avg:.4f})\n'.format(name=name, loss=value)
            print(print_string)
    return losses.avg  
        
def validate(val_loader, tmodel, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    meter_dict = {}
    for name in build_names():
        meter_dict[name] = AverageMeter()
    meter_dict['max_ht'] = AverageMeter()
    meter_dict['min_ht'] = AverageMeter()    
    meter_dict['max_paf'] = AverageMeter()    
    meter_dict['min_paf'] = AverageMeter()
    # switch to train mode
    model.eval()

    end = time.time()
    for i, (img, heatmap_target_total, heat_mask, paf_target, paf_mask) in enumerate(val_loader):
    #for i, (img, heatmap_target, heat_mask, paf_target, paf_mask) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        img = img.cuda()
        # heatmap_target = heatmap_target.cuda()
        heatmap_target_total_CUDA = []
        heatmap_target_total_CUDA.append(heatmap_target_total[0].cuda())
        heatmap_target_total_CUDA.append(heatmap_target_total[1].cuda())
        heatmap_target_total_CUDA.append(heatmap_target_total[2].cuda())
        heatmap_target_total_CUDA.append(heatmap_target_total[3].cuda())

        heat_mask = heat_mask.cuda()
        paf_target = paf_target.cuda()
        paf_mask = paf_mask.cuda()
        
        # compute output
        _,saved_for_loss = model(img)

        # 教师网络
        NetOutput, _ = tmodel(img)
        NetPAF = NetOutput[-2]
        NetHeatmap = NetOutput[-1]


        # total_loss, saved_for_log = get_loss(saved_for_loss, heatmap_target, heat_mask,
        #        paf_target, paf_mask)
        total_loss, saved_for_log = get_loss(saved_for_loss, heatmap_target_total_CUDA, heat_mask,
                                            paf_target, paf_mask, NetHeatmap, NetPAF)
               
        # for name,_ in meter_dict.items():
        #    meter_dict[name].update(saved_for_log[name], img.size(0))
            
        losses.update(total_loss.item(), img.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()  
        if i % args.print_freq == 0:
            print_string = 'Epoch: [{0}][{1}/{2}]\t'.format(epoch, i, len(val_loader))
            print_string +='Data time {data_time.val:.3f} ({data_time.avg:.3f})\t'.format( data_time=data_time)
            print_string += 'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(loss=losses)

            for name, value in meter_dict.items():
                print_string+='{name}: {loss.val:.4f} ({loss.avg:.4f})\t'.format(name=name, loss=value)
            print(print_string)
                
    return losses.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


print("Loading dataset...")
# load data
#num_workers=8
###-----------------加载训练数据-----------------###
train_data = get_loader(args.json_path, args.data_dir,
                        args.mask_dir, 368, 8,
                        'vgg', args.batch_size, params_transform = params_transform,
                        shuffle=True, training=True, num_workers=4)
print('train dataset len: {}'.format(len(train_data.dataset)))

# validation data
#valid_data = get_loader(args.json_path, args.data_dir, args.mask_dir, 368,
#                            8, preprocess='vgg', training=False,
#                            batch_size=args.batch_size, params_transform = params_transform, shuffle=False, #num_workers=4)
valid_data = get_loader(args.json_path, args.data_dir, args.mask_dir, 368,
                            8, preprocess='vgg', training=False,
                            batch_size=12, params_transform = params_transform, shuffle=False, num_workers=4)
print('val dataset len: {}'.format(len(valid_data.dataset)))
###-----------------结束-----------------###


# 获取教师网络, tmodel
tmodel = get_model(trunk='vgg19') #网络结构
#model = encoding.nn.DataParallelModel(model, device_ids=args.gpu_ids)
tmodel = torch.nn.DataParallel(tmodel).cuda() #使用GPU（cuda）
#加载教师网络预训练模型
tmodel.load_state_dict(torch.load(args.tmodel_path_file))
tmodel.eval()
#print(tmodel)

#设置学生网络, model
model = get_mobileNet_model()
#model = Net_other()
#model = get_model_res(trunk='vgg19')
model = torch.nn.DataParallel(model).cuda() #使用多GPU（cuda）
#print(model)

#加载学生网络的预训练模型
use_MobileNet(model, '/home/data/wangchenliang/pytorch-paf-model/ori_model/mobilenet_sgd_rmsprop_69.526.tar')
#use_SEResNet(model, args.model_path, 'SEResNet')

#保存网路初始化权重，保证网络可以复现, 训练中断后恢复需要注释掉
torch.save(model.state_dict(), args.init_model_param)

#固定加载预训练模型后的网络参数
###-----------------start(重头训练不注释，中断开始需要注释)-----------------###
#print(model.module.model0)
# for param in model.module.model0.parameters():
#     param.requires_grad = False
# trainable_vars = [param for param in model.parameters() if param.requires_grad]
#print(len(trainable_vars))
###-----------------end-----------------###

###---start---### #重头训练不注释，中断开始需要注释
# optimizer = torch.optim.SGD(trainable_vars, lr=args.lr,
#                           momentum=args.momentum,
#                           weight_decay=args.weight_decay,
#                           nesterov=args.nesterov)#优化方法
###---end---###

writer = SummaryWriter(log_dir=args.logdir)

###-----------------start(train)-----------------###
# for epoch in range(5):
#    #train for one epoch
#    train_loss = train(train_data, tmodel, model, optimizer, epoch)
#    #train_loss = 1
#
#    # evaluate on validation set
#    val_loss = validate(valid_data, tmodel, model, epoch)
#    #val_loss = 1
#
#    #print(epoch)
#
#    writer.add_scalars('data/scalar_group', {'train loss': train_loss,
#                                             'val loss': val_loss}, epoch)
###-----------------end(train)-----------------###

#Release all weights
###-----------------start(训练网络所有参数)-----------------###
for param in model.module.parameters():
    param.requires_grad = True
trainable_vars = [param for param in model.parameters() if param.requires_grad]
###-----------------end(训练网络所有参数)-----------------###

###-----------------start(设定优化方法,及策略)-----------------###
optimizer = torch.optim.SGD(trainable_vars, lr=args.lr,
                           momentum=args.momentum,
                           weight_decay=args.weight_decay,
                           nesterov=args.nesterov)

lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=3, min_lr=0, eps=1e-08)
###-----------------end(设定优化方法,及策略)-----------------###

best_val_loss = np.inf

# #model_save_filename = './network/weight/best_pose.pth'
#最终模型保存位置
model_save_filename = '/home/data/wangchenliang/pytorch-paf-model/pretrained-MobileNet-okhm-kd/best_pose-pretrained-MobileNet.pth'
#前五个epoch产生的模型保存位置
model_save_filename1 = '/home/data/wangchenliang/pytorch-paf-model/pretrained-MobileNet-okhm-kd/best_pose-pretrained-MobileNet-epoch5.pth'
#模型参数保存位置
model_all_para_save_filename = '/home/data/wangchenliang/pytorch-paf-model/pretrained-MobileNet-okhm-kd/model-save-para/model_all_para.pth'

#保存前五个epoch产生的模型
torch.save(model.state_dict(), model_save_filename1)

#加载网络模型参数
###-----------------start-----------------###
epoch_para = 5
load_train_model_para = True
if load_train_model_para == True:
    model_CKPT = torch.load(model_all_para_save_filename)
    #model_CKPT1 = torch.load(model_save_filename)
    model.load_state_dict(model_CKPT['state_dict'])
    optimizer.load_state_dict(model_CKPT['optimizer'])
    lr_scheduler.load_state_dict(model_CKPT['lr_scheduler'])
    #print(lr_scheduler)
    epoch_para = model_CKPT['epoch']
    best_val_loss = model_CKPT['best_val_loss']
###-----------------end-----------------###

best_val_loss = np.inf #后期中断可以取消这个注释
#
# # print(epoch_para)
# # print(optimizer)
# # print(best_val_loss)
# #for epoch in range(5, args.epochs):
for epoch in range(epoch_para, args.epochs):
#for epoch in range(0, 3):
    # train for one epoch
    train_loss = train(train_data, tmodel, model, optimizer, epoch)

    #print(epoch)
    #train_loss = 1

    # evaluate on validation set
    val_loss = validate(valid_data, tmodel, model, epoch)
    #val_loss = 1

    writer.add_scalars('data/scalar_group', {'train loss': train_loss,
                                             'val loss': val_loss}, epoch)
    lr_scheduler.step(val_loss)

    is_best = val_loss<best_val_loss
    best_val_loss = min(val_loss, best_val_loss)
    #is_best = False

    # 保存网络中间产生的所有参数
    torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_val_loss': best_val_loss, 'optimizer': optimizer.state_dict(), 'lr_scheduler': lr_scheduler.state_dict()}, model_all_para_save_filename)
    #torch.save({'state_dict': model.state_dict()}, model_all_para_save_filename)


    if is_best:
        torch.save(model.state_dict(), model_save_filename)

#writer.export_scalars_to_json(os.path.join(args.model_path,"tensorboard/all_scalars.json"))
writer.export_scalars_to_json(os.path.join('/home/data/wangchenliang/pytorch-paf-model/pretrained-MobileNet-okhm-kd',"all_scalars.json"))
writer.close()

#修改log信息位置需要更改选项行为 26:--logdir 321:writer = SummaryWriter(log_dir=args.logdir) 370:writer.export_scalars_to_json
