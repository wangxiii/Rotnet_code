"""Define a generic class for training and testing learning algorithms."""
from __future__ import print_function
import os
import os.path
import imp
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim

import utils
import datetime
import logging
import cv2
import matplotlib.pyplot as plt
from pdb import set_trace as breakpoint
import numpy as np
import pandas as pd
Test_DIR='./data_example/Nimstim_hsf_color_64/Nimstim_hsf_color_64'

class Algorithm():
    def __init__(self, opt):
        self.set_experiment_dir(opt['exp_dir'])
        self.set_log_file_handler()

        self.logger.info('Algorithm options %s' % opt)
       # print("DEBUG: print opt",opt)
        self.opt = opt
        self.init_all_networks()
        self.init_all_criterions()
        self.allocate_tensors()
        self.curr_epoch = 0
        self.optimizers = {}

        self.keep_best_model_metric_name = opt['best_metric'] if ('best_metric' in opt) else None

    def set_experiment_dir(self,directory_path):
        self.exp_dir = directory_path
        if (not os.path.isdir(self.exp_dir)):
            os.makedirs(self.exp_dir)

        self.vis_dir = os.path.join(directory_path,'visuals')
        if (not os.path.isdir(self.vis_dir)):
            os.makedirs(self.vis_dir)

        self.preds_dir = os.path.join(directory_path,'preds')
        if (not os.path.isdir(self.preds_dir)):
            os.makedirs(self.preds_dir)

    def set_log_file_handler(self):
        self.logger = logging.getLogger(__name__)

        strHandler = logging.StreamHandler()
        formatter = logging.Formatter(
                '%(asctime)s - %(name)-8s - %(levelname)-6s - %(message)s')
        strHandler.setFormatter(formatter)
        self.logger.addHandler(strHandler)
        self.logger.setLevel(logging.INFO)

        log_dir = os.path.join(self.exp_dir, 'logs')
        if (not os.path.isdir(log_dir)):
            os.makedirs(log_dir)

        now_str = datetime.datetime.now().__str__().replace(' ','_')
        now_str = now_str.replace(':', '_')
        self.log_file = os.path.join(log_dir, 'LOG_INFO_'+now_str+'.txt')
        self.log_fileHandler = logging.FileHandler(self.log_file)
        self.log_fileHandler.setFormatter(formatter)
        self.logger.addHandler(self.log_fileHandler)

    def init_all_networks(self):
        networks_defs = self.opt['networks']
        self.networks = {}
        self.optim_params = {}

        for key, val in networks_defs.items():
            self.logger.info('Set network %s' % key)
            def_file = val['def_file']
            net_opt = val['opt']
            self.optim_params[key] = val['optim_params'] if ('optim_params' in val) else None
            pretrained_path = val['pretrained'] if ('pretrained' in val) else None
            self.networks[key] = self.init_network(def_file, net_opt, pretrained_path, key)

    def init_network(self, net_def_file, net_opt, pretrained_path, key):
        self.logger.info('==> Initiliaze network %s from file %s with opts: %s' % (key, net_def_file, net_opt))
        if (not os.path.isfile(net_def_file)):
            raise ValueError('Non existing file: {0}'.format(net_def_file))
        network = imp.load_source("",net_def_file).create_model(net_opt)
        if pretrained_path != None:
            self.load_pretrained(network, pretrained_path)

        return network

    def load_pretrained(self, network, pretrained_path):
        self.logger.info('==> Load pretrained parameters from file %s:' % (pretrained_path))

        assert(os.path.isfile(pretrained_path))
        pretrained_model = torch.load(pretrained_path, map_location=torch.device('cpu'))
        if pretrained_model['network'].keys() == network.state_dict().keys():
            network.load_state_dict(pretrained_model['network'])
        else:
            self.logger.info('==> WARNING: network parameters in pre-trained file %s do not strictly match' % (pretrained_path))
            for pname, param in network.named_parameters():
                if pname in pretrained_model['network']:
                    self.logger.info('==> Copying parameter %s from file %s' % (pname, pretrained_path))
                    param.data.copy_(pretrained_model['network'][pname])

    def init_all_optimizers(self):
        self.optimizers = {}

        for key, oparams in self.optim_params.items():
            self.optimizers[key] = None
            if oparams != None:
                self.optimizers[key] = self.init_optimizer(
                        self.networks[key], oparams, key)
      #  print("DEBUG: optimizer is:",self.optimizers)
    def init_optimizer(self, net, optim_opts, key):
        optim_type = optim_opts['optim_type']
        learning_rate = optim_opts['lr']
        optimizer = None
        parameters = filter(lambda p: p.requires_grad, net.parameters())
        self.logger.info('Initialize optimizer: %s with params: %s for netwotk: %s'
            % (optim_type, optim_opts, key))
        if optim_type == 'adam':
            optimizer = torch.optim.Adam(parameters, lr=learning_rate,
                        betas=optim_opts['beta'])
        elif optim_type == 'sgd':
            optimizer = torch.optim.SGD(parameters, lr=learning_rate,
                momentum=optim_opts['momentum'],
                nesterov=optim_opts['nesterov'] if ('nesterov' in optim_opts) else False,
                weight_decay=optim_opts['weight_decay'])
        else:
            raise ValueError('Not supported or recognized optim_type', optim_type)

        return optimizer

    def init_all_criterions(self):
        criterions_defs = self.opt['criterions']
        self.criterions = {}
        for key, val in criterions_defs.items():
            crit_type = val['ctype']
            crit_opt = val['opt'] if ('opt' in val) else None
            self.logger.info('Initialize criterion[%s]: %s with options: %s' % (key, crit_type, crit_opt))
            self.criterions[key] = self.init_criterion(crit_type, crit_opt)

    def init_criterion(self, ctype, copt):
        return getattr(nn, ctype)(copt)

    def load_to_gpu(self):
        for key, net in self.networks.items():
            self.networks[key] = net.cuda()

        for key, criterion in self.criterions.items():
            self.criterions[key] = criterion.cuda()

        for key, tensor in self.tensors.items():
            self.tensors[key] = tensor.cuda()

    def save_checkpoint(self, epoch, suffix=''):
        for key, net in self.networks.items():
            if self.optimizers[key] == None: continue
            self.save_network(key, epoch, suffix=suffix)
            self.save_optimizer(key, epoch, suffix=suffix)

    def load_checkpoint(self, epoch, train=True, suffix=''):
        self.logger.info('Load checkpoint of epoch %d' % (epoch))

        for key, net in self.networks.items(): # Load networks
            if self.optim_params[key] == None: continue
            self.load_network(key, epoch,suffix)

        if train: # initialize and load optimizers
            self.init_all_optimizers()
            for key, net in self.networks.items():
                if self.optim_params[key] == None: continue
                self.load_optimizer(key, epoch,suffix)

        self.curr_epoch = epoch

    def delete_checkpoint(self, epoch, suffix=''):
        for key, net in self.networks.items():
            if self.optimizers[key] == None: continue

            filename_net = self._get_net_checkpoint_filename(key, epoch)+suffix
            if os.path.isfile(filename_net): os.remove(filename_net)

            filename_optim = self._get_optim_checkpoint_filename(key, epoch)+suffix
            if os.path.isfile(filename_optim): os.remove(filename_optim)

    def save_network(self, net_key, epoch, suffix=''):
        assert(net_key in self.networks)
        filename = self._get_net_checkpoint_filename(net_key, epoch)+suffix
        state = {'epoch': epoch,'network': self.networks[net_key].state_dict()}
        torch.save(state, filename)

    def save_optimizer(self, net_key, epoch, suffix=''):
        assert(net_key in self.optimizers)
        filename = self._get_optim_checkpoint_filename(net_key, epoch)+suffix
        state = {'epoch': epoch,'optimizer': self.optimizers[net_key].state_dict()}
        torch.save(state, filename)

    def load_network(self, net_key, epoch,suffix=''):
        assert(net_key in self.networks)
        filename = self._get_net_checkpoint_filename(net_key, epoch)+suffix
        assert(os.path.isfile(filename))
        if os.path.isfile(filename):
            checkpoint = torch.load(filename)
            self.networks[net_key].load_state_dict(checkpoint['network'])

    def load_optimizer(self, net_key, epoch,suffix=''):
        assert(net_key in self.optimizers)
        filename = self._get_optim_checkpoint_filename(net_key, epoch)+suffix
        assert(os.path.isfile(filename))
        if os.path.isfile(filename):
            checkpoint = torch.load(filename)
            self.optimizers[net_key].load_state_dict(checkpoint['optimizer'])

    def _get_net_checkpoint_filename(self, net_key, epoch):
        return os.path.join(self.exp_dir, net_key+'_net_epoch'+str(epoch))

    def _get_optim_checkpoint_filename(self, net_key, epoch):
        return os.path.join(self.exp_dir, net_key+'_optim_epoch'+str(epoch))

    def solve(self, data_loader_train, data_loader_test):
        self.max_num_epochs = self.opt['max_num_epochs']
        start_epoch = self.curr_epoch
        if len(self.optimizers) == 0:
            self.init_all_optimizers()

        eval_stats = {}
        train_stats = {}
        tprec = []
        tprec_c1 = []
        tprec_c2 = []
        tprec_c3 = []
        tprec_c4 = []
        tloss = []
        eprec = []
        eprec_c1 = []
        eprec_c2 = []
        eprec_c3 = []
        eprec_c4 = []
        eloss = []
        self.init_record_of_best_model()
        for self.curr_epoch in range(start_epoch, self.max_num_epochs):
            self.logger.info('Training epoch [%3d / %3d]' % (self.curr_epoch + 1, self.max_num_epochs))
            self.adjust_learning_rates(self.curr_epoch)
            train_stats = self.run_train_epoch(data_loader_train, self.curr_epoch)
            self.logger.info('==> Training stats: %s' % (train_stats))
            tprec.append(train_stats['prec1'])
            tloss.append(train_stats['loss'])
            tprec_c1.append(train_stats['prec_c1'])
            tprec_c2.append(train_stats['prec_c2'])
            tprec_c3.append(train_stats['prec_c3'])
            tprec_c4.append(train_stats['prec_c4'])
            self.save_checkpoint(self.curr_epoch + 1)  # create a checkpoint in the current epoch
            if start_epoch != self.curr_epoch:  # delete the checkpoint of the previous epoch
                self.delete_checkpoint(self.curr_epoch)

            if data_loader_test is not None:
                eval_stats = self.evaluate(data_loader_test)
                self.logger.info('==> Evaluation stats: %s' % (eval_stats))
                self.keep_record_of_best_model(eval_stats, self.curr_epoch)
                """
                loggerx.add_scalar("evaluate loss", eval_stats['loss'], global_step=self.curr_epoch)
                # 添加第二条日志：正确率-全局迭代次数
                loggerx.add_scalar("test accuary", eval_stats['prec1'], global_step=self.curr_epoch)
               # loggerx.add_image("train image sample", data_loader_test, global_step=self.curr_epoch)
               """
                eprec.append(eval_stats['prec1'])
                eloss.append(eval_stats['loss'])
                eprec_c1.append(eval_stats['prec_c1'])
                eprec_c2.append(eval_stats['prec_c2'])
                eprec_c3.append(eval_stats['prec_c3'])
                eprec_c4.append(eval_stats['prec_c4'])
        if data_loader_test is  None:
            data = {"tprec": tprec, "tloss": tloss, "tprec_c1": tprec_c1, "tprec_c2": tprec_c2, "tprec_c3": tprec_c3,
                    "tprec_c4": tprec_c4,
                    }
        else:
            data = {"tprec": tprec, "tloss": tloss, "tprec_c1": tprec_c1, "tprec_c2": tprec_c2, "tprec_c3": tprec_c3,
                "tprec_c4": tprec_c4,
                "eprec": eprec, "eloss": eloss, "eprec_c1": eprec_c1, "eprec_c2": eprec_c2, "eprec_c3": eprec_c3,
                "eprec_c4": eprec_c4}
        file = "1.csv"  # 保存文件位置，即当前工作路径下的csv文件
        # print("DEBUG: file is",file)

        data = pd.DataFrame(data)  # 要保存的数据
        #  print("DEBUG: data is ",data)
        data.to_csv(file)  # 数据写入，index=False表示不加索引
        print("DEBUG: after writing")
        self.print_eval_stats_of_best_model()

    def run_train_epoch(self, data_loader, epoch):
        self.logger.info('Training: %s' % os.path.basename(self.exp_dir))
        self.dloader       = data_loader
        self.dataset_train = data_loader.dataset

        for key, network in self.networks.items():
            #这是在干嘛，在eval模式下和optimizer=none有什么关系？
            if self.optimizers[key] == None: network.eval()
            else: network.train()

        disp_step   = self.opt['disp_step'] if ('disp_step' in self.opt) else 50
        train_stats = utils.DAverageMeter()
        self.bnumber = len(data_loader())
        ####这里有一个问题，关键这句话我就看不懂..
        for idx, batch in enumerate(tqdm(data_loader(epoch))):
            self.biter = idx
            train_stats_this = self.train_step(batch)
            train_stats.update(train_stats_this)
            #if (idx+1) % disp_step == 0:

                #self.logger.info('==> Iteration [%3d][%4d / %4d]: %s' % (epoch+1, idx+1, len(data_loader), train_stats.average()))

        return train_stats.average()

    def evaluate(self, dloader,visual_heap=False):
        self.logger.info('Evaluating: %s' % os.path.basename(self.exp_dir))

        self.dloader = dloader
        self.dataset_eval = dloader.dataset
        test_file=os.listdir(Test_DIR)
     #   print("DEBUG,len(dloader):",len(dloader()))
        self.logger.info('==> Dataset: %s [%d images]' % (dloader.dataset.name, len(dloader())))
        for key, network in self.networks.items():
            network.eval()

        eval_stats = utils.DAverageMeter()
        self.bnumber = len(dloader())
        for idx, batch in enumerate(tqdm(dloader())):
            self.biter = idx
            eval_stats_this = self.evaluation_step(batch)
            #eval_stats.update(eval_stats_this) batch平均，useless when batch =1
            if (visual_heap):
               # print("DEBUG: name ",Test_DIR+'/'+test_file[idx])
                img=cv2.imread(Test_DIR+'/'+test_file[idx])

                #print("DEBUG: imgs shape",img)
                heatmap=eval_stats_this['heat']#只在evaluate模式中返回heatmap
                #print("DEBUG: img origin", img)
                heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
                heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
                superimposed_img = heatmap * 0.4 + img  # 这里的0.4是热力图强度因子
                #print("DEBUG: idx is",idx)

                cv2.imwrite("data_example/heatmap/Nimstim_hsf"+'/'+str(idx)+'.jpeg', superimposed_img)  # 将图像保存到硬盘
            if(not visual_heap):
                self.logger.info('==> Results: %s' % eval_stats_this)
        if(not visual_heap):
            self.logger.info('==> Results: %s' % eval_stats.average())
            return eval_stats.average()
        return 0

    def adjust_learning_rates(self, epoch):
        # filter out the networks that are not trainable and that do
        # not have a learning rate Look Up Table (LUT_lr) in their optim_params
        optim_params_filtered = {k:v for k,v in self.optim_params.items()
            if (v != None and ('LUT_lr' in v))}

        for key, oparams in optim_params_filtered.items():
            LUT = oparams['LUT_lr']
            lr = next((lr for (max_epoch, lr) in LUT if max_epoch>epoch), LUT[-1][1])
            self.logger.info('==> Set to %s optimizer lr = %.10f' % (key, lr))
            for param_group in self.optimizers[key].param_groups:
                param_group['lr'] = lr

    def init_record_of_best_model(self):
        self.max_metric_val = None
        self.best_stats = None
        self.best_epoch = None

    def keep_record_of_best_model(self, eval_stats, current_epoch):
        if self.keep_best_model_metric_name is not None:

            if (self.keep_best_model_metric_name not in eval_stats):
                raise ValueError('The provided metric {0} for keeping the best model is not computed by the evaluation routine.'.format(metric_name))
            metric_val = eval_stats[self.keep_best_model_metric_name]
            if self.max_metric_val is None or metric_val > self.max_metric_val:
                self.max_metric_val = metric_val
                self.best_stats = eval_stats
                self.save_checkpoint(self.curr_epoch+1, suffix='.best')
                if self.best_epoch is not None:
                    self.delete_checkpoint(self.best_epoch+1, suffix='.best')
                self.best_epoch = current_epoch
                self.print_eval_stats_of_best_model()

    def print_eval_stats_of_best_model(self):
        if self.best_stats is not None:
            metric_name = self.keep_best_model_metric_name
            self.logger.info('==> Best results w.r.t. %s metric: epoch: %d - %s' % (metric_name, self.best_epoch+1, self.best_stats))


    # FROM HERE ON ARE ABSTRACT FUNCTIONS THAT MUST BE IMPLEMENTED BY THE CLASS
    # THAT INHERITS THE Algorithms CLASS
    def train_step(self, batch):
        """Implements a training step that includes:
            * Forward a batch through the network(s)
            * Compute loss(es)
            * Backward propagation through the networks
            * Apply optimization step(s)
            * Return a dictionary with the computed losses and any other desired
                stats. The key names on the dictionary can be arbitrary.
        """
        pass

    def evaluation_step(self, batch):
        """Implements an evaluation step that includes:
            * Forward a batch through the network(s)
            * Compute loss(es) or any other evaluation metrics.
            * Return a dictionary with the computed losses the evaluation
                metrics for that batch. The key names on the dictionary can be
                arbitrary.
        """
        pass

    def allocate_tensors(self):
        """(Optional) allocate torch tensors that could potentially be used in
            in the train_step() or evaluation_step() functions. If the
            load_to_gpu() function is called then those tensors will be moved to
            the gpu device.
        """
        self.tensors = {}
