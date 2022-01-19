from __future__ import print_function
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import os
import torchnet as tnt
import utils
import PIL
import pickle
from tqdm import tqdm
import time
"""
import sys
sys.path.append("")
from NetworkInNetwork import 
"""
from . import Algorithm
from pdb import set_trace as breakpoint

def heatmap(features,output,modelname='NIN'):
#def draw_CAM(model, img_path, save_path, transform=None, visual_heatmap=False):
    # 为了能读取到中间梯度定义的辅助函数
    # 预测得分最高的那一类对应的输出score

    def extract(g):
        global features_grad
        features_grad = g

    _, pred = output.topk(1, 1, True, True)
    pred_class = pred.t()
#这种情况还有待改进
    if (modelname == 'Alexnet'):
        features.register_hook(extract)
        pred_class.backward()  # 计算梯度
        grads = features_grad  # 获取梯度
        pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))
        pooled_grads = pooled_grads[0]
        for i in range(512):
            features[0][i, ...] *= pooled_grads[i, ...]
    # 此处batch size默认为1，所以去掉了第0维（batch size维）

    #print("DEBUG: feature's shape is,",features.shape)
    features = features[0]
    # 512是最后一层feature的通道数


    # 以下部分同Keras版实现
    heatmap = features.detach().numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    return heatmap

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# 分开计算4个类别准确率
def accuracy2(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    batch_size_real = batch_size / 4
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    correctc = list(0. for i in range(4))
    res = []
    # 分类准确率计算，默认不能有topk>1情况
    for label_idx in range(len(target)):
        label_single = target[label_idx]
        # print("DEBUG: single correct is:",correct[0][label_idx])
        correctc[label_single] += correct[0][label_idx].item()
    correctc = [i * 100 / batch_size_real for i in correctc]
    res.append(correctc)

    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class ClassificationModel(Algorithm):
    def __init__(self, opt):
        Algorithm.__init__(self, opt)
        print("DEBUG: optimizer in ClassificationModel: ", self.optimizers)

    def allocate_tensors(self):
        self.tensors = {}
        self.tensors['dataX'] = torch.FloatTensor()
        self.tensors['labels'] = torch.LongTensor()

    def train_step(self, batch):
        return self.process_batch(batch, do_train=True)

    def evaluation_step(self, batch):

        return self.process_batch(batch, do_train=False)

    def process_batch(self, batch, do_train=True, visual_heap=False):
        # *************** LOAD BATCH (AND MOVE IT TO GPU) ********
        start = time.time()
        modelname= 'NIN' #不同model对应不同heatmap计算方法
        self.tensors['dataX'].resize_(batch[0].size()).copy_(batch[0])
        self.tensors['labels'].resize_(batch[1].size()).copy_(batch[1])
        dataX = self.tensors['dataX']
        labels = self.tensors['labels']
        batch_load_time = time.time() - start

        # ********************************************************

        # ********************************************************
        start = time.time()
        if do_train:  # zero the gradients
            self.optimizers['model'].zero_grad()
        # ********************************************************

        # ***************** SET TORCH VARIABLES ******************
        dataX_var = torch.autograd.Variable(dataX, volatile=(not do_train))
        labels_var = torch.autograd.Variable(labels, requires_grad=False)
        # ********************************************************
        #修改成只输出全局层or分类器以前的feature

        feature =0
        if modelname =='NIN'and visual_heap:
            feat1=self.networks['model']._feature_blocks[0](dataX_var)
            feat2 = self.networks['model']._feature_blocks[1](feat1)
            feat3 = self.networks['model']._feature_blocks[2](feat2)
            feature = self.networks['model']._feature_blocks[3](feat3)
            #print("DEBUG: feats after block is,",feat1)
        # ************ FORWARD THROUGH NET ***********************
        pred_var = self.networks['model'](dataX_var)
        # ********************************************************

        # *************** COMPUTE LOSSES *************************
        record = {}
        loss_total = self.criterions['loss'](pred_var, labels_var)
        record['prec1'] = accuracy2(pred_var.data, labels, topk=(1,))[1].item()
        record['loss'] = loss_total.item()
        record['prec_c1'] = accuracy2(pred_var.data, labels, topk=(1,))[0][0]
        record['prec_c2'] = accuracy2(pred_var.data, labels, topk=(1,))[0][1]
        record['prec_c3'] = accuracy2(pred_var.data, labels, topk=(1,))[0][2]
        record['prec_c4'] = accuracy2(pred_var.data, labels, topk=(1,))[0][3]
        if visual_heap:
            record['heat'] = heatmap(feature,pred_var,modelname)
            #record['dataX']=dataX_var
        # ********************************************************
        # 可视化原始热力图

        # ****** BACKPROPAGATE AND APPLY OPTIMIZATION STEP *******
        if do_train:
            loss_total.backward()
            self.optimizers['model'].step()
        # ********************************************************
        batch_process_time = time.time() - start
        total_time = batch_process_time + batch_load_time
        record['load_time'] = 100 * (batch_load_time / total_time)
        record['process_time'] = 100 * (batch_process_time / total_time)

        return record
